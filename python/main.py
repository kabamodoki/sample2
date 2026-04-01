import re
import sqlite3

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

def init_db():
    conn = sqlite3.connect("data.db")
    conn.execute("DROP TABLE IF EXISTS snippets;")
    # 日本語2文字以上を確実に拾うための設定
    conn.execute("CREATE VIRTUAL TABLE snippets USING fts5(content, tokenize='unicode61 categories 2');")
    
    target_data = [
        "氏名の形式は「山田 太郎」と空白を入力できる形式です",
        "生年月日は「yyyy mm dd」と入力できる形式です",
        "申請フォームは企業の入力情報が必要です"
    ]
    dummy_data = [f"ダミーデータ_{i:02}: 開発仕様書セクション{i}" for i in range(20)]
    
    all_data = [(d,) for d in target_data + dummy_data]
    conn.executemany("INSERT INTO snippets(content) VALUES (?)", all_data)
    conn.commit()
    conn.close()

init_db()

class QueryRequest(BaseModel):
    prompt: str

@app.post("/rag-debug")
async def rag_debug(request: QueryRequest):
    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    
    # 1. 検索キーワードの抽出
    ignore_chars = r'[とがをに、。を作成したい。入って。る]'
    clean_text = re.sub(ignore_chars, ' ', request.prompt)
    keywords = list(set([w for w in clean_text.split() if len(w) >= 2]))
    
    hit_statistics = []
    all_hit_contents = []

    # 2. キーワードごとに「何がヒットしたか」を抽出
    for k in keywords:
        cursor.execute("SELECT content FROM snippets WHERE snippets MATCH ?", (f'"{k}"',))
        rows = cursor.fetchall()
        if rows:
            matched_texts = [r[0] for r in rows]
            hit_statistics.append({
                "keyword": k,
                "matched_documents": matched_texts  # なにがヒットしたか文章を出す
            })
            all_hit_contents.extend(matched_texts)

    # 3. 全体での検索実行（上位10件をリスト形式で取得）
    search_query = " OR ".join([f'"{k}"' for k in keywords]) if keywords else f'"{request.prompt}"'
    cursor.execute("SELECT content FROM snippets WHERE snippets MATCH ? ORDER BY rank LIMIT 10", (search_query,))
    final_rows = cursor.fetchall()
    
    # 4. 重複を排除しつつリスト化
    sent_context_list = [r[0] for r in final_rows]
    
    conn.close()

    # JSONレスポンスの構築
    return {
        "status": "success",
        "match_count": len(all_hit_contents),     # ヒットした延べ回数
        "hit_word_types": len(hit_statistics),    # ヒットした単語の種類数
        "hit_statistics": hit_statistics,         # キーワードごとの詳細
        "sent_context": sent_context_list         # ヒットした文章のリスト
    }