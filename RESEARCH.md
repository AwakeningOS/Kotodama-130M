# Kotodama-130M current research record — 2026-08-02

この文書は `kotodama_stable_loop_130m_v2` だけを対象にします。
過去のモデル定義や実験は現行判断の根拠に使いません。

## 結論

現行モデルは、KDA/MLA 3:1ハイブリッド、安定対角ループ、
SwiGLU、QK正規化付き潜在MLAキャッシュを組み合わせます。1億の最初の
評価時点までにT=8経路も学習されるよう、深度ランプは1Bトークン、
T=8の固定tailは1/16とします。

## 文献で埋まった問い

### 反復構造と安定性

[Parcae](https://arxiv.org/abs/2604.12946) は、負の対角連続系を離散化した
安定な状態注入、正規化した入力の反復注入、系列単位の可変深度学習を示しています。
Kotodamaの `StableDiagonalInjection` とprelude正規化はこの系統です。

[Scaling by Thinking in Continuous Space](https://arxiv.org/abs/2502.05171) は、
prelude/core/coda構造、各反復での入力注入、ランダムな初期状態、可変反復深度を
大規模学習で使用しています。現行のlabelled trainingで使うlike-init状態は
この既知設計に基づきます。

Parcaeは、学習時の平均反復深度が推論時スケーリングの上限を決めることも報告して
います。このため、初期段階から推論深度T=8を一定確率で学習させます。

### KDA/MLAハイブリッド

[Kimi Linear](https://arxiv.org/abs/2510.26692) は、KDAとglobal MLAを3:1で
組み合わせたハイブリッドを大規模に検証しています。Kotodamaは小型・denseですが、
KDA/MLA比率とstrict NoPE MLAの根拠はこの系統にあります。

### QK正規化付き潜在KVキャッシュ

[QK-Normed MLA](https://arxiv.org/abs/2606.16310) は、post-projection QK RMSNormを
維持したま、静的なkey gainをquery側へ吸収し、動的なinverse-RMS scalarを
latentと共に保存すれば潜在decodeできることを示しています。現行実装の
永続cacheは128要素のlatentと12要素のinverse-RMS、合計140要素/tokenです。

### DeepLoopを採用しない理由

[DeepLoop](https://arxiv.org/abs/2607.13491) はPost-LN DeepNorm型の共有残差に対する
初期化則です。KotodamaはPre-RMSNorm blockとParcae型注入を使っており、式の前提が
異なるため、その係数をそのまま移植していません。

## 固定した実装判断

| 項目 | 現行値 |
|---|---|
| architecture ID | `kotodama_stable_loop_130m_v2` |
| depth ramp | 1B tokens |
| T=8 early exposure | 常時約1/16以上 |
| loop initial state | labelled trainingのみlike-init乱数 |
| FFN | SwiGLU |
| MLA persistent cache | 128 latent + 12 inverse-RMS |
| EOD cache reset | 全階層、batch row単位 |
| first stop | 100M tokens |
| production optimizer | AdamW |

## 文献だけでは埋まらない問い

以下は実際の学習と評価でしか分かりません。

- KDA/MLAとParcae型ループを組み合わせた130Mモデルが、固定深度の競合モデルより
  良い言語性能を出すか。
- 1/16のT=8 tailと1B rampが、このGPU予算で品質と速度の良い折衷になるか。
- T=2からT=8へ増やしたとき、検訿lossと実ベンチが改善するか。
- like-init学習後、決定的なzero-init評価で性能が落ちないか。

これらのモデル固有診断は100Mと500Mで確認し、正式な三モデル共通ベンチは1Bで
Deltaxis、KaiNomosと同じ凍結問題集を使って実施します。複数seedや
厳密なFLOP一致は必須条件とせず、同じ実用ベンチでモデル一式を競わせます。
