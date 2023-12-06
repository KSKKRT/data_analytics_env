# data-analytics-env

### ディレクトリ構成

```.
|-- pyproject.toml
|-- requirements-dev.lock : 開発環境のlockファイル
|-- requirements.lock : 環境全体のlockファイル
|-- README.md
|-- model : 機械学習モデルの格納先
|-- features : 特徴量生成ファイル、生成した特徴用の保存先
|-- input : コンペデータの格納先
|-- notebook : ノートブック格納先
|-- output : 実行結果の格納先
|-- utils
`-- yamls : hydra で使うための config ファイル群
```

### 環境構築
```sh
rye sync
```
