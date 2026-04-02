package com.example.rag;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
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
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import jakarta.annotation.PostConstruct;

/**
 * 検索とLLM連携を統括するコントローラー
 */
@RestController
@RequestMapping("/rag-debug")
public class RagController {

	private final Directory index = new ByteBuffersDirectory();
	private final JapaneseAnalyzer analyzer = new JapaneseAnalyzer();
	private final Tika tika = new Tika();

	@Autowired
	private LlmIntegrationService llmService;

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
	 * RAG検索を実行し、結果をLLMサービスへ渡してレスポンスを返却する
	 */
	@PostMapping
	public Map<String, Object> search(@RequestBody Map<String, String> request) throws Exception {
		String prompt = request.getOrDefault("prompt", "");

		// 1. ベクトル生成（16次元）
		float[] queryVector = mockEmbedding(prompt);

		try (IndexReader reader = DirectoryReader.open(index)) {
			IndexSearcher searcher = new IndexSearcher(reader);

			// 2. ハイブリッド検索の実行
			TopDocs topDocs = searcher.search(buildHybridQuery(prompt, queryVector), MAX_RESULTS);

			// 3. 検索結果（マニュアル抜粋）のリスト化
			List<String> contexts = new ArrayList<>();
			for (ScoreDoc sd : topDocs.scoreDocs) {
				Document d = searcher.doc(sd.doc);
				contexts.add("[" + d.get("category") + "] " + d.get("content").replace("\n", " "));
			}

			// 4. LLM連携サービスを呼び出し、回答（またはデバッグ用案内）を取得
			String llmAnswer = llmService.processRequest(prompt, contexts);

			// 5. APIレスポンスの構築
			Map<String, Object> response = new LinkedHashMap<>();
			response.put("status", "success");
			response.put("input_prompt", prompt);
			response.put("match_count", topDocs.totalHits.value);

			// LLMからの回答（Localモード時は案内文）をセット
			response.put("llm_response", llmAnswer);

			return response;
		}
	}

	/**
	 * 検索結果からテキスト内容のみを抽出
	 */
	private List<String> extractContexts(IndexSearcher searcher, TopDocs topDocs) throws IOException {
		List<String> contexts = new ArrayList<>();
		for (ScoreDoc sd : topDocs.scoreDocs) {
			Document d = searcher.doc(sd.doc);
			contexts.add("[" + d.get("category") + "] " + d.get("content").replace("\n", " "));
		}
		return contexts;
	}

	private Query buildHybridQuery(String prompt, float[] vector) throws Exception {
		BooleanQuery.Builder builder = new BooleanQuery.Builder();
		builder.add(new KnnFloatVectorQuery("vector", vector, MAX_RESULTS), BooleanClause.Occur.SHOULD);
		String cleanQuery = prompt.replaceAll("[とがをに、。フォーム]", " ").trim();
		if (!cleanQuery.isEmpty()) {
			QueryParser parser = new QueryParser("content", analyzer);
			builder.add(parser.parse(cleanQuery), BooleanClause.Occur.SHOULD);
		}
		return builder.build();
	}

	private void loadTextResource(IndexWriter writer, String fileName) throws IOException {
		ClassPathResource resource = new ClassPathResource(fileName);
		if (!resource.exists())
			return;
		try (BufferedReader br = new BufferedReader(new InputStreamReader(resource.getInputStream(), StandardCharsets.UTF_8))) {
			br.lines()
					.map(line -> line.split(",", 2))
					.filter(parts -> parts.length >= 2)
					.forEach(parts -> addDocument(writer, parts[0].trim(), parts[1].trim()));
		}
	}

	private void loadPdfResource(IndexWriter writer, String fileName) throws Exception {
		ClassPathResource resource = new ClassPathResource(fileName);
		if (!resource.exists())
			return;
		String fullText = tika.parseToString(resource.getInputStream());
		List<String> chunks = splitIntoChunks(fullText, CHUNK_SIZE);
		for (String chunk : chunks) {
			addDocument(writer, "PDFマニュアル", chunk.trim());
		}
	}

	private void addDocument(IndexWriter writer, String category, String content) {
		try {
			Document doc = new Document();
			doc.add(new StringField("category", category, Field.Store.YES));
			doc.add(new TextField("content", content, Field.Store.YES));
			doc.add(new KnnFloatVectorField("vector", mockEmbedding(content), VectorSimilarityFunction.COSINE));
			writer.addDocument(doc);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private float[] mockEmbedding(String text) {
		float[] v = new float[VECTOR_DIM];
		v[3] = calc(text, 0.85f, "セットアップ", "募集期間", "締切", "補助金名");
		v[4] = calc(text, 0.85f, "フロー", "プロセス", "ステップ", "手続き");
		v[8] = calc(text, 0.95f, "差し戻し", "不備", "修正依頼", "再申請");
		v[0] = calc(text, 0.50f, "gBizID", "二要素", "パスワード", "ログイン", "認証");
		v[11] = calc(text, 0.65f, "CSV", "ダウンロード", "抽出", "ZIP");
		return v;
	}

	private float calc(String text, float weight, String... keys) {
		float score = 0.0f;
		for (String key : keys) {
			if (text.contains(key))
				score += weight;
		}
		return Math.min(score, 1.0f);
	}

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
		if (sb.length() > 0)
			chunks.add(sb.toString().trim());
		return chunks;
	}
}