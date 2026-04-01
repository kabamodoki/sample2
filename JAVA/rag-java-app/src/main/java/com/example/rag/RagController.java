package com.example.rag;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
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
import org.apache.tika.Tika;
import org.springframework.core.io.ClassPathResource;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import jakarta.annotation.PostConstruct;

/**
 * Jグランツ操作マニュアル特化型 RAG検索コントローラー
 * 16次元ベクトル検索とキーワード検索を組み合わせ、コンソールに詳細ログを出力します。
 */
@RestController
@RequestMapping("/rag-debug")
public class RagController {

    private final Directory index = new ByteBuffersDirectory();
    private final JapaneseAnalyzer analyzer = new JapaneseAnalyzer();
    private final Tika tika = new Tika();

    // Jグランツの業務フローに最適化した16次元定義
    private static final int VECTOR_DIM = 16;
    private static final int MAX_RESULTS = 10;
    private static final int CHUNK_SIZE = 300;

    @PostConstruct
    public void init() throws Exception {
        try (IndexWriter writer = new IndexWriter(index, new IndexWriterConfig(analyzer))) {
            loadTextResource(writer, "sample_data.txt");
            loadPdfResource(writer, "操作マニュアル_事務局管理者用.pdf");
        }
    }

    /**
     * テキストリソースのインデックス登録
     */
    private void loadTextResource(IndexWriter writer, String fileName) throws IOException {
        ClassPathResource resource = new ClassPathResource(fileName);
        if (!resource.exists()) return;

        try (BufferedReader br = new BufferedReader(new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8))) {
            br.lines()
              .map(line -> line.split(",", 2))
              .filter(parts -> parts.length >= 2)
              .forEach(parts -> addDocument(writer, parts[0].trim(), parts[1].trim()));
            System.out.println("--- [SYSTEM] テキストデータの登録完了 ---");
        }
    }

    /**
     * PDFリソースの解析とチャンク分割登録
     */
    private void loadPdfResource(IndexWriter writer, String fileName) throws Exception {
        ClassPathResource resource = new ClassPathResource(fileName);
        if (!resource.exists()) {
            System.err.println("[ERROR] PDF未検出: " + fileName);
            return;
        }

        System.out.println("--- [SYSTEM] PDF解析開始: " + fileName + " ---");
        String fullText = tika.parseToString(resource.getInputStream());
        List<String> chunks = splitIntoChunks(fullText, CHUNK_SIZE);

        for (String chunk : chunks) {
            addDocument(writer, "PDFマニュアル", chunk.trim());
        }
        System.out.println("--- [SYSTEM] PDF登録完了: " + chunks.size() + " chunks ---");
    }

    /**
     * ドキュメントの共通登録処理
     */
    private void addDocument(IndexWriter writer, String category, String content) {
        try {
            Document doc = new Document();
            doc.add(new StringField("category", category, Field.Store.YES));
            doc.add(new TextField("content", content, Field.Store.YES));

            float[] vector = mockEmbedding(content);
            doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));

            writer.addDocument(doc);
        } catch (IOException e) {
            System.err.println("[ERROR] ドキュメント追加失敗: " + e.getMessage());
        }
    }

    /**
     * 検索実行
     */
    @PostMapping
    public Map<String, Object> search(@RequestBody Map<String, String> request) throws Exception {
        String prompt = request.getOrDefault("prompt", "");
        float[] queryVector = mockEmbedding(prompt);

        System.out.println("\n==================================================");
        System.out.println(" [SEARCH START]");
        System.out.println(" プロンプト: " + prompt);
        System.out.println(" 検索ベクトル: " + Arrays.toString(queryVector));
        System.out.println("==================================================");

        try (IndexReader reader = DirectoryReader.open(index)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            TopDocs topDocs = searcher.search(buildHybridQuery(prompt, queryVector), MAX_RESULTS);
            
            return createResponseAndLog(searcher, topDocs);
        }
    }

    /**
     * ハイブリッドクエリ構築
     */
    private Query buildHybridQuery(String prompt, float[] vector) throws Exception {
        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        
        // ベクトル検索
        builder.add(new KnnFloatVectorQuery("vector", vector, MAX_RESULTS), BooleanClause.Occur.SHOULD);

        // キーワード検索
        String cleanQuery = prompt.replaceAll("[とがをに、。フォーム]", " ").trim();
        if (!cleanQuery.isEmpty()) {
            QueryParser parser = new QueryParser("content", analyzer);
            builder.add(parser.parse(cleanQuery), BooleanClause.Occur.SHOULD);
        }
        return builder.build();
    }

    /**
     * 詳細ログ出力とレスポンス構築
     */
    private Map<String, Object> createResponseAndLog(IndexSearcher searcher, TopDocs topDocs) throws IOException {
        List<String> contexts = new ArrayList<>();
        System.out.println(" ヒット総数: " + topDocs.totalHits.value);
        System.out.println("--------------------------------------------------");

        int rank = 1;
        for (ScoreDoc sd : topDocs.scoreDocs) {
            Document d = searcher.doc(sd.doc);
            String cat = d.get("category");
            String txt = d.get("content").replace("\n", " ");

            contexts.add("[" + cat + "] " + txt);
            
            // コンソールに詳細を全量出力
            System.out.printf("[%d位] スコア: %.4f | カテゴリ: %s%n", rank++, sd.score, cat);
            System.out.println("内容: " + txt);
            System.out.println("--------------------------------------------------");
        }
        System.out.println(" [SEARCH END]\n");

        Map<String, Object> res = new LinkedHashMap<>();
        res.put("match_count", topDocs.totalHits.value);
        res.put("sent_context", contexts);
        return res;
    }

    /**
     * Jグランツ実務に合わせた重み付け擬似Embedding
     */
    private float[] mockEmbedding(String text) {
        float[] v = new float[VECTOR_DIM];

        // 申請・審査系（重要：重み大 0.8-0.95）
        v[3] = calc(text, 0.85f, "セットアップ", "募集期間", "締切", "補助金名");
        v[4] = calc(text, 0.85f, "フロー", "プロセス", "ステップ", "手続き");
        v[5] = calc(text, 0.80f, "項目", "入力欄", "テンプレート", "必須");
        v[7] = calc(text, 0.90f, "承認", "決裁", "受付", "審査開始");
        v[8] = calc(text, 0.95f, "差し戻し", "不備", "修正依頼", "再申請");
        v[9] = calc(text, 0.95f, "却下", "辞退", "取下げ", "終了");

        // アカウント・通知・データ系（重み中 0.5-0.65）
        v[0] = calc(text, 0.50f, "gBizID", "二要素", "パスワード", "ログイン", "認証");
        v[1] = calc(text, 0.60f, "権限", "付与", "事務局管理者", "担当者", "アカウント");
        v[2] = calc(text, 0.60f, "制度", "予算", "公募", "詳細情報");
        v[6] = calc(text, 0.60f, "添付", "ファイル", "PDF", "JPG", "容量");
        v[10] = calc(text, 0.60f, "通知", "メール", "自動送信", "文面");
        v[11] = calc(text, 0.65f, "CSV", "ダウンロード", "抽出", "ZIP");
        v[13] = calc(text, 0.50f, "事業者画面", "表示", "プレビュー", "マイページ");
        v[14] = calc(text, 0.60f, "採択", "交付決定", "確定", "実施報告");

        // ヘルプ・環境系（重み低 0.25-0.3）
        v[12] = calc(text, 0.30f, "ヘルプデスク", "問合せ", "トラブル", "FAQ");
        v[15] = calc(text, 0.25f, "メンテナンス", "ブラウザ", "OS", "環境");

        return v;
    }

    private float calc(String text, float weight, String... keys) {
        float score = 0.0f;
        for (String key : keys) {
            if (text.contains(key)) score += weight;
        }
        return Math.min(score, 1.0f);
    }

    /**
     * チャンク分割
     */
    private List<String> splitIntoChunks(String text, int max) {
        String clean = text.replaceAll("(?m)^[ \t]*\r?\n", "\n").replaceAll("\n+", "\n");
        List<String> chunks = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (String line : clean.split("\n")) {
            if (sb.length() + line.length() > max) {
                chunks.add(sb.toString().trim());
                sb.setLength(0);
            }
            sb.append(line).append(" ");
        }
        if (sb.length() > 0) chunks.add(sb.toString().trim());
        return chunks;
    }
}