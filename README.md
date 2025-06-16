# 資料集說明

本資料庫以不同刑事案件類型整理，提供訓練與評估文字搜尋或問答系統所需的文件、查詢與關聯標註。

## 資料夾結構

```
.
├── data
│   ├── forgery
│   ├── fraud
│   ├── larceny
│   ├── sexoffences
│   └── snatch
```

每個犯罪類型資料夾均包含下列檔案：

- `*_judgment_summary.json`：案件摘要。
- `sample_50.json` / `sample_500.json`：示例資料。
- `format/`：主要資料，內含
  - `corpus.json`：文件內容。
  - `queries.json`：查詢句。
  - `qrels.json`：查詢與文件的相關標註。

## 範例

`queries.json` 中的內容範例如下：

```json
{
  "id": 0,
  "text": "被告冒用兄名義偽造私文書，以避警察發現酒後駕駛，坦承犯行並表現悔悟。"
}
```

`qrels.json` 的標註範例：

```json
{
  "qid": 0,
  "docid": 349,
  "relevance": 1
}
```

`corpus.json` 中的文件範例：

```json
{
  "id": 0,
  "text": "臺灣臺中地方法院刑事簡易判決104年度審簡字第896號公訴人臺灣臺中地方法院檢察署檢察官被告游麗華上列被告因偽造文書案件，經檢察官提起"
}
```

## 使用方式

使用者可根據 `queries.json` 提供的查詢與 `qrels.json` 的相關度標註，在 `corpus.json` 建立索引或進行檢索實驗。各類型資料夾提供的結構一致，可依需求選擇特定犯罪類型進行研究。


## BM25 檢索範例

檢索流程分為離線索引建立與線上查詢兩步

```bash
# 建立索引
python build_bm25_index.py data/fraud fraud_index.json
# 讀取索引執行查詢
python bm25_retrieval.py fraud_index.json "被告明知詐欺集團成員" 3
```

上述指令會先在 `data/fraud` 產生 `fraud_index.json`，再以該索引取得前 3 筆相似文件編號與分數。


## 虛擬環境 (uv)

本專案支援使用 [uv](https://github.com/astral-sh/uv) 建立 Python 虛擬環境。首次使用時可透過下列指令建立並安裝依賴：

```bash
# 安裝 uv (若尚未安裝)
pip install uv
# 在專案根目錄建立 .venv 環境
uv venv .venv
# 安裝依賴 (本專案目前僅使用標準函式庫，無需額外套件)
uv pip install -r requirements.txt
```

完成後即可在 `.venv` 環境中執行範例程式。

## MCP 伺服器

本資料庫提供 `mcp_server.py` 作為簡易 [MCP](https://github.com/UDICatNCHU/mcp) 伺服器實作，
透過 BM25 搜尋工具回應查詢。此程式需 Python 3.10 以上版本才能順利執行，
可在虛擬環境啟用後以下列指令啟動：

 ```bash
 uv run mcp install mcp_server.py
 ```

此指令會回傳伺服器狀態確認訊息。

伺服器新增 `expand_search` 工具，可利用 Gemini 擴充查詢後再執行 BM25 搜尋，
預設使用 `gemini-2.0-flash` 模型。建議先使用 `search` 觀察檢索結果，
若結果不足再呼叫 `expand_search`。呼叫時需提供 `query` 與可選的 `top_k` 參數。

此外，`read_fraud_data` 工具會直接回傳資料列表，建議搭配
`offset` 與 `limit` 參數分批取得結果，以避免一次回傳過多內容造成解析問題。例如：

```bash
{"tool": "read_fraud_data", "args": {"offset": 0, "limit": 20}}
```

## Gemini MCP 客戶端

若要使用 `gemini_mcp_client.py` 啟動智能助手，請先設定 Google Gemini API 金鑰。建議在專案根目錄建立 `.env` 檔並填入：

```bash
echo "GEMINI_API_KEY=你的金鑰" > .env
```

或直接匯出環境變數：

```bash
export GEMINI_API_KEY=你的金鑰
```

完成設定後即可啟動客戶端：

```bash
python gemini_mcp_client.py
```

`.env` 檔已加入 `.gitignore`，不會被納入版本控管。

## 簡易網頁介面


`web_server.py` 提供一個基於 Flask 的聊天頁面，可與 Gemini 智能助手互動。啟動前請先設定 `GEMINI_API_KEY` 環境變數。

```bash
# 安裝相依套件
pip install -r requirements.txt
# 設定金鑰並啟動伺服器
export GEMINI_API_KEY=your_key
python web_server.py
```

啟動後瀏覽 <http://localhost:8000/> 即可進行對話，詢問資料集或搜尋相關問題。

