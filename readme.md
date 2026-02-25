# Twicome

Twitch VOD コメントを収集し、検索・分析・可視化するための Web アプリケーションです。  
データ収集バッチ、MySQL 取り込み、FAISS 類似検索、コミュニティノート生成までを一体で運用できます。

## 重要な運用前提（必読）

- 更新系エンドポイント（例: `/add_user`, `/like`, `/dislike`, FAISS 更新系 API）は、インターネットへ直接公開する設計にはなっていません。
- 外部公開する場合は、以下の要素を検討してください。
  - アプリ層の認証・認可（更新系 API の保護）
  - レート制限（特に更新系・検索系 API）
  - 監査ログ/アクセスログの整備
  - ネットワーク境界（WAF/許可IP/内部APIの閉域化）の明確化

## 主な機能

- Twitch VOD コメントの自動収集（Twitch API + TwitchDownloaderCLI）
- コメント一覧 UI（フィルタ、並び替え、無限スクロール、リアクション）
- 類似検索（FAISS + SentenceTransformer）
- 典型度検索（重心距離）
- 感情スライダー検索（感情アンカーの加重合成）
- コミュニティノート表示と危険度スコア表示
- 統計ページ（時間帯、曜日、配信者別活動率、影響度分析）
- コメント当てクイズ

## システム構成

- `batch/scripts/get_vod_list_batch.py`:
  - 監視対象ユーザの VOD 一覧を Twitch API から取得
- `batch/scripts/batch_download_comments.py`:
  - `library/TwitchDownloaderCLI` で VOD コメント JSON を取得
- `batch/scripts/insertdb.py`:
  - コメント JSON を `users` / `vods` / `comments` に upsert
- `batch/scripts/build_faiss_index.py`:
  - コメント埋め込みを生成し、ユーザ別 FAISS インデックスを作成
- `batch/scripts/generate_community_notes.py`:
  - dislike 閾値以上のコメントに OpenRouter 経由でノートを生成
- `app/`:
  - FastAPI + Jinja2 で検索 UI / API / 統計 UI を提供

## ディレクトリ概要

- `app/`: Web アプリ本体（FastAPI, router, template）
- `migrate/`: Alembic マイグレーション
- `batch/scripts/`: バッチ本体
- `batch/prompts/`: コミュニティノート生成プロンプト
- `util/`: Twitch トークン更新、ユーザ ID 取得などの補助スクリプト
- `data/`: 環境別の入出力データ（`development`, `honban` など）
- `library/TwitchDownloaderCLI`: コメント取得用 CLI バイナリ
- `dbschema.md`: MySQL スキーマ

## セットアップ

### 1. 前提

- Python 3.11 系
- MySQL 8 系
- Twitch API の `CLIENT_ID` / `CLIENT_SECRET` / `ACCESS_TOKEN`
- 収集を行う場合は `library/TwitchDownloaderCLI` が実行可能であること

必要なら実行権限を付与します。

```bash
chmod +x library/TwitchDownloaderCLI
```

### 2. Python 環境

`run_batch.sh` はデフォルトで `.venv/bin/python` を使うため、ルートに仮想環境を作る運用が前提です。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r app/requirements.txt -r batch/requirements.txt
```

### 3. 設定ファイル

`.env.example` と `faiss_config.json.example` をコピーして編集します。

```bash
cp .env.example .env
cp faiss_config.json.example faiss_config.json
cp batch/prompts/community_note_system_prompt.txt.example batch/prompts/community_note_system_prompt.txt
```

最低限、次の項目は設定してください。

- `DATABASE_URL`:
  - Web アプリ（SQLAlchemy）用
- `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`:
  - バッチ（mysql-connector-python）用
- `ACCESS_TOKEN`, `CLIENT_ID`:
  - Twitch API 用
- `FAISS_CONFIG_PATH`:
  - 既定は `./faiss_config.json`
- `OPENROUTER_API_KEY`:
  - コミュニティノート生成を使う場合のみ必須

補助設定（任意）:

- `ROOT_PATH`（既定 `/twicome`）
- `DEFAULT_LOGIN`
- `QUICK_LINK_LOGINS`
- `HOST_CHECK_ENABLED`
- `APP_ENV`, `BATCH_DATA_DIR` などのバッチ入出力先

Docker Compose で動かす場合、DB 接続先はコンテナ名 `db:3306` を使います（ホストへの DB ポート公開は不要）。

### 4. MySQL マイグレーション適用

初期スキーマは Alembic で管理します。  
（`dbschema.md` は参照用、実際の適用はマイグレーションを使用）

```bash
docker compose run --rm migrate
```

### 5. 監視対象ユーザ CSV

バッチは `TARGET_USERS_CSV` を読みます。  
CSV には `name,id` 列が必要です（例: `targetusers.csv`）。

## 実行方法

### バッチ一括実行

```bash
./run_batch.sh
```

別 env ファイルを使う場合:

```bash
./run_batch.sh --env-file .env.development
```

FAISS インデックス作成をスキップする場合:

```bash
SKIP_FAISS=1 ./run_batch.sh
```

バッチの実行順序:

1. `get_vod_list_batch.py`
2. `batch_download_comments.py`
3. `insertdb.py`
4. `build_faiss_index.py`（`SKIP_FAISS!=1` のとき）
5. `generate_community_notes.py`（失敗しても非致命）

ログは既定で `data/${APP_ENV}/logs/batch_YYYYmmdd_HHMMSS.log` に出力されます。

`OPENROUTER_API_KEY` 未設定時は、`generate_community_notes.py` が警告終了します（他工程は成功扱い）。

### 既存コメント JSON だけ取り込み

```bash
./run_import_comments.sh --env-file .env.development --comments-dir data/honban/comments
```

既存 VOD を再取り込みしたい場合:

```bash
./run_import_comments.sh --env-file .env.development --comments-dir data/honban/comments --reingest-existing-vods
```

### Web アプリ起動（ローカル）

```bash
cd app
../.venv/bin/uvicorn main:app --env-file ../.env --host 0.0.0.0 --port 8000 --reload
```

`ROOT_PATH=/twicome` の場合:

- `http://localhost:8000/twicome`

`ROOT_PATH=/` の場合:

- `http://localhost:8000/`

### Docker Compose 起動

本番寄り設定（`db` + `migrate` + `app` + `batch` を同時起動）:

```bash
docker compose up --build
```

- `db`: ホストへは公開しない（`app`/`batch`/`migrate` から `db:3306` で接続）
- `app`: `http://localhost:8000/twicome`
- `batch`: 起動直後に1回実行、その後4時間ごとに定期実行
- `migrate`: 起動時に `alembic upgrade head` を実行（完了後に停止）

開発設定（`db` + `migrate` + `app` + `batch` を同時起動）:

```bash
docker compose -f docker-compose.dev.yml up --build
```

- `db`: ホストへは公開しない（`app`/`batch`/`migrate` から `db:3306` で接続）
- `app`: `http://localhost:8011/`

`batch` / `library` のコード変更を反映するには `batch` イメージ再ビルド（`docker compose build batch`）が必要です。

`docker-compose*.yml` のボリュームは `./data/...` を参照するため、運用に合わせてパスを調整してください。

ホストから DB に直接入りたい場合は、ポート公開ではなく `docker compose exec` を使ってください。

```bash
docker compose exec db mysql -uappuser -p appdb
```

外部 DB を使いたい場合は、`COMPOSE_DATABASE_URL` と `COMPOSE_MYSQL_*` を上書きしてください。

例:

```bash
COMPOSE_DATABASE_URL="mysql+pymysql://appuser:apppass@host.docker.internal:3306/appdb?charset=utf8mb4" docker compose up --build
```

## CI
<!-- 
`.github/workflows/ci.yml` で以下を自動実行します。

1. MySQL 8 起動
2. `alembic upgrade head`
3. 必須テーブル（`users`, `vods`, `comments`, `community_notes`, `vod_ingest_markers`）の存在確認 -->

## Web 画面

- `/`: トップ（ユーザ選択、人気コメント、導線）
- `/u/{login}`: コメント一覧
- `/u/{login}/stats`: 統計
- `/u/{login}/quiz`: コメント当てクイズ
- `/add_user`: 監視対象追加
- `/manual`: 使い方ガイド

注意:

- `/add_user` は `/host/targetusers.csv` に書き込む実装です。
- Docker では `targetusers.csv` を `/host/targetusers.csv` にマウントして使ってください。

## 主要 API

- `GET /api/u/{login}`: コメント一覧
- `GET /api/u/{login}/similar`: 類似検索
- `GET /api/u/{login}/centroid`: 典型度検索
- `GET /api/u/{login}/emotion`: 感情検索
- `GET /api/emotion_axes`: 感情軸一覧
- `POST /like/{comment_id}?count=1`: いいね加算
- `POST /dislike/{comment_id}?count=1`: わるいね加算
- `GET /api/u/{login}/quiz/start`: クイズ問題生成

## FAISS 検索について

- モデルは `hotchpotch/static-embedding-japanese` を使用
- `build_faiss_index.py` がユーザ別に `.faiss` / `.meta.json` を生成
- アプリ側は `app/faiss_data` を参照
- Docker 利用時は `app/faiss_data` へインデックスディレクトリをマウントして整合を取ってください

## 補助スクリプト（util）

- `util/refreshtoken.py`:
  - `.env` の `ACCESS_TOKEN` 自動更新
- `util/tokens.py`:
  - クライアントクレデンシャルで token 取得
- `util/userid.py`:
  - Twitch login から user id 取得
- `util/extract_twitch_comments.py`:
  - コメント JSON から特定ユーザコメントを抽出

## ライセンスと依存関係

- TwitchDownloaderCLI の著作権表示: `library/COPYRIGHT.txt`
- TwitchDownloaderCLI のサードパーティライセンス: `library/THIRD-PARTY-LICENSES.txt`
- Python 依存関係: `app/requirements.txt`, `batch/requirements.txt`

## 開発メモ

このプロジェクトは、Kilo / Claude Code / Codex CLI を含む LLM を活用して開発されています。
