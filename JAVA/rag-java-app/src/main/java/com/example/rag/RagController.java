package com.example.rag;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.analysis.ja.JapaneseAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import jakarta.annotation.PostConstruct;

/**
 * ハイブリッドRAG検索コントローラー
 * 従来のキーワード検索（Lucene）と意味ベースのベクトル検索（HNSW）を組み合わせた検索基盤を提供します。
 */
@RestController
@RequestMapping("/rag-debug")
public class RagController {

    // メモリ上にインデックスを保持するディレクトリ
    private final Directory index = new ByteBuffersDirectory();
    // 日本語の形態素解析器（Kuromoji）
    private final JapaneseAnalyzer analyzer = new JapaneseAnalyzer();
    
    // ベクトルの次元数（PoC用に8つの業務概念を定義）
    private static final int VECTOR_DIM = 8; 

    /**
     * アプリケーション起動時の初期化処理。
     * リソースファイルの読み込みとインデックス構築を開始します。
     */
    @PostConstruct
    public void init() throws Exception {
        loadAndIndexResources();
    }

    /**
     * 【仕組み：データインジェクション】
     * 外部テキストファイル（CSV形式）を読み込み、検索可能な形式に変換して保存します。
     * カテゴリ（完全一致用）、本文（全文検索用）、ベクトル（意味検索用）の3つの形式で保持します。
     */
    private void loadAndIndexResources() throws Exception {
        ClassPathResource resource = new ClassPathResource("sample_data.txt");
        
        try (BufferedReader br = new BufferedReader(new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8));
             IndexWriter writer = new IndexWriter(index, new IndexWriterConfig(analyzer))) {
            
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",", 2);
                if (parts.length < 2) continue;

                String category = parts[0].trim();
                String content = parts[1].trim();

                Document doc = new Document();
                // カテゴリ：フィルタリングや分類用
                doc.add(new StringField("category", category, Field.Store.YES));
                // 本文：キーワードマッチング用（形態素解析される）
                doc.add(new TextField("content", content, Field.Store.YES));
                
                // ベクトル：意味の近さを計算するための数値列（HNSWグラフを構築）
                float[] vector = mockEmbedding(content); 
                doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
                
                writer.addDocument(doc);
            }
            System.out.println("--- [SYSTEM] ハイブリッドインデックスの構築完了 ---");
        }
    }

    /**
     * 【仕組み：検索エントリポイント】
     * クライアントからの質問（プロンプト）を受け取り、ベクトル化、検索、結果の整形を一連の流れで行います。
     */
    @PostMapping
    public Map<String, Object> search(@RequestBody Map<String, String> request) throws Exception {
        String prompt = request.getOrDefault("prompt", "");
        float[] queryVector = mockEmbedding(prompt);

        System.out.println("\n--- [SEARCH START] ---");
        System.out.println("[INPUT] プロンプト: " + prompt);
        System.out.println("[PARAM] 検索ベクトル: " + Arrays.toString(queryVector));

        try (IndexReader reader = DirectoryReader.open(index)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            
            // ハイブリッドクエリを生成して実行
            Query hybridQuery = buildHybridQuery(prompt, queryVector);
            TopDocs topDocs = searcher.search(hybridQuery, 5);

            return buildResponse(searcher, topDocs);
        }
    }

    /**
     * 【仕組み：ハイブリッド評価ロジック】
     * 「意味の近さ（Vector）」と「単語の一致（Keyword）」を合成したクエリを作成します。
     * これにより、言い換え表現に強いベクトル検索と、固有名詞に強いキーワード検索の利点を両立させます。
     */
    private Query buildHybridQuery(String prompt, float[] queryVector) throws Exception {
        BooleanQuery.Builder chain = new BooleanQuery.Builder();

        // 1. ベクトル近傍探索（K-Nearest Neighbor）: 意味的に近い上位5件を候補にする
        KnnFloatVectorQuery knnQuery = new KnnFloatVectorQuery("vector", queryVector, 5);
        chain.add(new BooleanClause(knnQuery, BooleanClause.Occur.SHOULD));

        // 2. テキストマッチング: 形態素解析後の重要語句が含まれるか評価する（BM25アルゴリズム）
        String keywordQueryStr = prompt.replaceAll("[とがをに、。フォーム]", " ");
        QueryParser parser = new QueryParser("content", analyzer);
        Query keywordQuery = parser.parse(keywordQueryStr);
        chain.add(new BooleanClause(keywordQuery, BooleanClause.Occur.SHOULD));

        return chain.build();
    }

    /**
     * 【仕組み：テキストのベクトル化（Embedding）】
     * 自然文をAIが理解できる数値列に変換します（PoC用モック）。
     * 実際の実装では、ここでOpenAI等のAPIを呼び出し、高次元（例：1536次元）のベクトルを取得します。
     */
    private float[] mockEmbedding(String text) {
        float[] v = new float[VECTOR_DIM];
        
        // 特定の業務概念（次元）に対応するキーワードが含まれる場合、その次元の数値を高める
        if (matches(text, "氏名", "名前", "本人", "性別", "年齢")) v[0] = 0.9f;
        if (matches(text, "メール", "電話", "アドレス", "連絡")) v[1] = 0.9f;
        if (matches(text, "住所", "郵便", "居住", "ビル", "番地")) v[2] = 0.9f;
        if (matches(text, "申請", "承認", "提出", "受理", "戻し")) v[3] = 0.9f;
        if (matches(text, "金融", "銀行", "口座", "振込", "カード", "クレジット")) v[4] = 0.9f;
        if (matches(text, "勤務", "出勤", "退勤", "雇用", "社員", "休憩")) v[5] = 0.9f;
        if (matches(text, "休暇", "有給", "休み", "慶弔")) v[6] = 0.9f;
        if (matches(text, "交通費", "精算", "ルート", "税金", "源泉", "扶養")) v[7] = 0.9f;

        return v;
    }

    private boolean matches(String text, String... keywords) {
        return Arrays.stream(keywords).anyMatch(text::contains);
    }

    /**
     * 【仕組み：レスポンス構築】
     * Luceneが算出した合算スコアに基づき、上位の結果をAPIレスポンスとして返却します。
     * ここで返却された text が RAG の文脈（Context）としてLLMに渡されることになります。
     */
    private Map<String, Object> buildResponse(IndexSearcher searcher, TopDocs topDocs) throws Exception {
        List<Map<String, Object>> statistics = new ArrayList<>();
        List<String> contexts = new ArrayList<>();

        System.out.println("[RESULT] ヒット件数: " + topDocs.totalHits.value);

        int rank = 1;
        for (ScoreDoc sd : topDocs.scoreDocs) {
            Document d = searcher.doc(sd.doc);
            String category = d.get("category");
            String content = d.get("content");
            
            contexts.add("[" + category + "] " + content);

            Map<String, Object> stat = new HashMap<>();
            stat.put("category", category);
            stat.put("text", content);
            stat.put("total_score", sd.score);
            statistics.add(stat);
            
            System.out.printf("  順位 %d: [%s] スコア: %.4f / 内容: %s%n", rank++, category, sd.score, content);
        }

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("match_count", topDocs.totalHits.value);
        response.put("hit_statistics", statistics);
        response.put("sent_context", contexts);
        return response;
    }
}