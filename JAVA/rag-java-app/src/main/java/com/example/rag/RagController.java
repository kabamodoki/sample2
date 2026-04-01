package com.example.rag;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.lucene.analysis.ja.JapaneseAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import jakarta.annotation.PostConstruct;

@RestController
@RequestMapping("/rag-debug")
public class RagController {

    @Autowired
    private JdbcTemplate jdbcTemplate;

    private final Directory index = new ByteBuffersDirectory();
    private final JapaneseAnalyzer analyzer = new JapaneseAnalyzer();

    @PostConstruct
    public void init() throws Exception {
        prepareMasterData();
        syncToLuceneIndex();
    }

    private void prepareMasterData() {
        jdbcTemplate.execute("DROP TABLE IF EXISTS master_snippets");
        jdbcTemplate.execute("CREATE TABLE master_snippets (id INTEGER PRIMARY KEY, content TEXT)");

        String[] data = {
            "氏名の形式は「山田 太郎」と空白を入力できる形式です",
            "生年月日は「yyyy mm dd」と入力できる形式です",
            "申請フォームは企業の入力情報が必要です"
        };
        for (String s : data) {
            jdbcTemplate.update("INSERT INTO master_snippets (content) VALUES (?)", s);
        }
    }

    private void syncToLuceneIndex() throws Exception {
        List<String> allContent = jdbcTemplate.queryForList("SELECT content FROM master_snippets", String.class);
        
        try (IndexWriter writer = new IndexWriter(index, new IndexWriterConfig(analyzer))) {
            for (String content : allContent) {
                Document doc = new Document();
                doc.add(new TextField("content", content, Field.Store.YES));
                writer.addDocument(doc);
            }
        }
        System.out.println("--- [SYSTEM] SQLiteデータをLuceneインデックスへ同期しました ---");
    }

    @PostMapping
    public Map<String, Object> search(@RequestBody Map<String, String> request) throws Exception {
        String prompt = request.getOrDefault("prompt", "");
        
        // 入力ログ
        System.out.println("\n[INPUT] ユーザー入力プロンプト: " + prompt);
        
        try (IndexReader reader = DirectoryReader.open(index)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            
            // クエリ構築と検索ログ
            String luceneQuery = buildSearchQuery(prompt);
            System.out.println("[QUERY] 構築されたLuceneクエリ: " + luceneQuery);

            Query query = new QueryParser("content", analyzer).parse(luceneQuery);
            TopDocs topDocs = searcher.search(query, 10);

            System.out.println("[RESULT] ヒット件数: " + topDocs.totalHits.value);

            return buildResponse(searcher, topDocs);
        }
    }

    private String buildSearchQuery(String prompt) {
        String cleanText = prompt.replaceAll("[とがをに、。を作成したい。入って。るフォーム]", " ");
        List<String> keywords = Arrays.stream(cleanText.split("\\s+"))
                                      .filter(word -> word.length() >= 2)
                                      .map(word -> "\"" + word + "\"")
                                      .collect(Collectors.toList());

        return keywords.isEmpty() ? prompt : String.join(" OR ", keywords);
    }

    private Map<String, Object> buildResponse(IndexSearcher searcher, TopDocs topDocs) throws Exception {
        List<Map<String, Object>> statistics = new ArrayList<>();
        List<String> contexts = new ArrayList<>();

        for (ScoreDoc sd : topDocs.scoreDocs) {
            String text = searcher.doc(sd.doc).get("content");
            contexts.add(text);

            Map<String, Object> stat = new HashMap<>();
            stat.put("text", text);
            stat.put("score", sd.score);
            statistics.add(stat);
            
            // ヒット詳細ログ
            System.out.println("  - ヒット内容: " + text + " (Score: " + sd.score + ")");
        }

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("status", "success");
        response.put("match_count", topDocs.totalHits.value);
        response.put("hit_statistics", statistics);
        response.put("sent_context", contexts);
        return response;
    }
}