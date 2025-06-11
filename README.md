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

