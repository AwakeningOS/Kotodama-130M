# Kotodama-130M

Kotodama-130Mは、同じ小さな思考回路を何度か繰り返して使う、約1.3億パラメータの
実験的な言語モデルです。学習済みweight、tokenizer、学習データは同梱していません。
このrepositoryには、モデル本体、学習、checkpoint再開、データ準備、文章生成のコードだけを
収録しています。自分の文章を用意し、自分の小さな言語モデルを育てて遊ぶためのものです。

> 現在はbase model用です。ChatGPTのような対話調整済みモデルではありません。
> 最初は質問応答より、文章の続きを生成させる使い方が向いています。

## どんな仕組みか

普通の言語モデルは、異なる処理層を上から順番に一度ずつ通ります。Kotodamaは、最初に
文章を読み取る部分と、最後に出力を整える部分の間に、何度も使い回す共通回路を置きます。

```text
入力文
  ↓
最初の読み取り（KDA → MLA）
  ↓
共通の反復回路（KDA×3 → MLA → KDA×3 → MLA）をT回
  ↓
最後の整形（KDA → KDA）
  ↓
次のtokenの確率
```

イメージとしては、文章を一度読んだあと、同じメモをT回読み直しながら少しずつ更新し、
最後に答えを書く構造です。回路の重みは共有されるため、Tを増やしてもモデルファイル自体は
大きくなりません。推論時のTは、品質と速度を交換するつまみになります。

### KDAとMLA

- **KDA**は、読んだ内容を固定サイズの状態へ少しずつ書き込む記憶です。文章が長くなっても、
  保存する状態が際限なく増えません。局所的な流れや更新の追跡を担当します。
- **MLA**は、過去のtokenを圧縮して保存し、必要な場所をまとめて参照します。KDAだけでは
  拾いにくい、離れた情報を補います。

KotodamaではKDAを3回使うごとにMLAを1回置きます。位置回転（RoPE）は使わず、順序は
因果的な処理と内部状態から学びます。

### 繰り返しても壊れにくくする工夫

同じ回路を何度も通すと、内部値が急に大きくなったり、入力を忘れたりしやすくなります。
そこで毎回、最初に読んだ入力を安定した割合で戻しながら状態を更新します。学習時はT=2〜8を
系列ごとに変え、浅い経路だけに依存しないよう最初から少数のT=8系列も混ぜます。

## 必要なもの

- Linux
- Python 3.12
- CUDA対応NVIDIA GPU（24GB級を推奨）
- 自分で利用権を確認したUTF-8の学習テキスト
- 自分で作る49,152語彙のSentencePiece tokenizer

CPUでもモデル構築と一部テスト、低速な生成はできます。本格学習はCUDA GPU向けです。

## インストール

```bash
git clone https://github.com/AwakeningOS/Kotodama-130M.git
cd Kotodama-130M
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0
python -m pip install -r requirements.txt
```

`flash-linear-attention`の対応CUDA/PyTorch環境が必要です。異なる環境へ移植する場合は、
まずCPUテストと短いGPU試験を行ってください。

## 1. tokenizerを作る

学習用テキストを1行1文書のUTF-8ファイルとして用意します。複数ファイルを渡せます。

```bash
python scripts/train_tokenizer.py \
  --input corpus/train.txt \
  --model-prefix tokenizer/kotodama
```

Kotodamaの語彙数は49,152で固定です。特殊tokenは`unk=0`、`bos=1`、`eos=2`、`pad=3`、
`<|eod|>=4`になります。小さすぎる文章集合では49,152語彙を作れないため、tokenizer学習には
十分な量と種類の文章を使ってください。

既存tokenizerを使う場合も、語彙数49,152と`<|eod|>`のID 4が必須です。

## 2. データを詰める

訓練用と検証用を分けて準備します。`.txt`は非空の1行を1文書、`.jsonl`は各行の`text`
フィールドを1文書として読みます。

```bash
python scripts/prepare_data.py \
  --train corpus/train.txt \
  --validation corpus/validation.txt \
  --tokenizer-model tokenizer/kotodama.model \
  --output-dir data/my-corpus
```

問題集、benchmark、正解表、他人の個人情報、利用許諾のない文章を学習へ混ぜないでください。

## 3. まず1 stepだけ動かす

1 optimizer stepは65,536 tokenです。RTX 3090 24GB向けの既定値はmicro batch 8です。

```bash
python train.py \
  --data-dir data/my-corpus \
  --run-dir runs/my-first-kotodama \
  --allow-gpu \
  --target-tokens 65536 \
  --max-steps 1
```

VRAMが足りない場合は`--micro-batch 4`、`2`、`1`を試してください。1 step全体のtoken数は
変わらず、内部の分割回数が増えます。最初のstepはコンパイルを含むため、通常より遅くなります。

## 4. 時間を区切って育てる

```bash
python train.py \
  --data-dir data/my-corpus \
  --run-dir runs/my-first-kotodama \
  --allow-gpu \
  --resume \
  --target-tokens 100000000 \
  --max-minutes 30
```

時間切れ、Ctrl+C、SIGTERMでは、現在のstepを終えてからcheckpointを保存します。通常checkpointは
最新2個を残します。同じコマンドへ`--resume`を付ければ、optimizer、乱数、データ位置を含めて
続きから再開します。

## 5. 文章の続きを生成する

```bash
python scripts/generate_text.py \
  --checkpoint runs/my-first-kotodama/step_0000001.pt \
  --tokenizer-model tokenizer/kotodama.model \
  --prompt "昔々、海のそばに" \
  --depth 2 \
  --max-new-tokens 128
```

`--depth`を1、2、4、8と変えると、同じ重みで計算量を変えられます。ただし深くすれば必ず
良くなるわけではありません。生成途中でdepthを変えるとcacheの履歴が一致しなくなるため、
depthは1回の生成中は固定されます。

## 速度を上げるための工夫

- 共通反復回路だけを`torch.compile`し、Tの違いによる全体再コンパイルを避けます。
- 同じTの系列をまとめ、終了済み系列に対する無駄な反復を減らします。
- 浅いTではactivation checkpointを省き、深いTだけ再計算を使ってVRAMを守ります。
- KDAは融合kernelを使い、MLAは圧縮cacheを保存します。
- 生成時は毎回全文を計算し直さず、KDA状態とMLA cacheを引き継ぎます。
- 文書境界がない通常経路では、大きな文書maskを作りません。

内部のRTX 3090測定では、反復回路だけのcompileがdepth 4で約62.8%高速化しました。生成は
prompt 1,024 / 新規256 tokenの条件で、T2が34.07 token/s、T8が10.25 token/sでした。
これは特定のCUDA・PyTorch・FLA環境と学習途中checkpointの測定であり、保証値ではありません。

## 学習時間の目安

RTX 3090 1枚、BF16、学習初期では約13,000 token/sを観測しました。単純計算なら100Mは
約2.1時間、500Mは約10.7時間、1Bは約21時間、16Bは約14日です。ただし学習が進むと平均Tが
増えて遅くなり、初回compile、検証、checkpoint、ストレージ速度も加わります。

実用的には次を大まかな予算として見てください。

| 学習量 | RTX 3090 1枚の目安 |
|---:|---:|
| 1 step（65,536 token） | 初回compile込みで数分以内 |
| 100M | 約2〜3時間 |
| 500M | 約12〜18時間 |
| 1B | 約1〜2日 |
| 16B | 約2〜4週間 |

データ、GPU、冷却、micro batch、CUDA環境で大きく変わります。最初から長期間回さず、1 step、
30分、100Mの順でcheckpointと生成を確認するのがおすすめです。

## 構成

```text
kotodama/                 モデル、cache、生成、データ読み込み
train.py                  AdamW事前学習と厳密resume
recommended_inference.py  現在の参考depth preset（T2）
scripts/train_tokenizer.py
scripts/prepare_data.py
scripts/generate_text.py
tests/                    CPU correctness tests
RESEARCH.md               設計根拠と未解決点
```

## 収録していないもの

- 学習済みweightとoptimizer checkpoint
- tokenizer
- 学習・検証データ
- 内部benchmark結果のraw log
- privateな実験・運用ファイル

checkpointは1個約780MBになるため、通常のGitへcommitしないでください。`.gitignore`で除外しています。

## 注意

Kotodamaは研究・学習用の実験実装です。品質、安全性、事実性は保証されません。公開前の
129,964,332パラメータ実装ではCPU/GPU correctness試験と短期学習を通していますが、学習済みweightを
配布しているわけではありません。自分のデータの権利、個人情報、生成物の利用については、利用者が
確認してください。

MIT License
