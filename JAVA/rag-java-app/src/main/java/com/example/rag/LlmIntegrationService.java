package com.example.rag;

import java.util.List;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class LlmIntegrationService {

	@Value("${LLM_API_ENDPOINT:https://api.example.com/v1/chat}")
	private String llmEndpoint;

	@Value("${MODE:production}")
	private String mode;

	/**
	 * RAGのコンテキストとユーザー入力を統合し、LLMまたはログへ出力
	 */
	public String processRequest(String userInput, List<String> contexts) {
		// システムプロンプトの取得
		String systemPrompt = buildSystemPrompt();

		// 検索結果（マニュアル）の整形
		String formattedContext = contexts.stream()
				.collect(Collectors.joining("\n---\n", "### System Context (Manual):\n", ""));

		// ユーザー入力セクションの整形
		String formattedUserInput = "### User Input:\n" + userInput;

		// 全メッセージの結合
		String finalMessage = String.format("%s\n\n%s\n\n%s",
				systemPrompt, formattedContext, formattedUserInput);

		// モード判定
		if ("local".equalsIgnoreCase(mode)) {
			printMaintenanceLog(finalMessage);
			return "【Local Mode】プロンプトをログに出力しました。内容を確認してください。";
		} else {
			return sendToLlm(finalMessage);
		}
	}

	private String buildSystemPrompt() {
	    return """
	            # Role
	            あなたはJグランツの実行判断エージェントです。
	            「System Context（マニュアル）」を根拠に「User Input（依頼）」を評価し、適切なアクションをとってください。

	            # Instructions
	            1. Tool Matching: 依頼を完遂できるMCPツールがあるか確認してください。
	            2. Validation: マニュアルのルールと照合し、不足や違反があれば実行せずユーザーに問い合せてください。
	            3. Execution & Response: 条件を満たした場合のみツールを実行し、以下の形式で回答してください。

	            # Output Format (回答形式)
	            実行したアクションの内容に応じて、必ず以下の情報を含めて回答してください。

	            - **作成・設定系アクションの場合:**
	              操作が成功した旨と、生成されたリソースの【アクセス用URL】を必ず提示してください。
	              (例: 「申請フォーム『AAA』を作成しました。以下のURLから確認・編集が可能です：[URL]」)

	            - **検索・参照系アクションの場合:**
	              「System Context」およびツールから得られた結果を整理し、ユーザーの疑問に対する【直接的な回答】を提示してください。
	              箇条書きを活用し、関連するマニュアルのセクション名も併記してください。

	            - **実行不能・情報不足の場合:**
	              何が足りないのか、またはなぜ実行できないのかを、マニュアルの規定を引用して具体的に説明してください。

	            # Constraints
	            - URLやIDなどの動的な情報は、必ずMCPツールの実行結果（Output）に基づいて出力してください。
	            - 推測でURLを生成しないでください。
	            
	            ---
	            """;
	}

	private void printMaintenanceLog(String message) {
		System.out.println("\n" + "=".repeat(30));
		System.out.println(" [MAINTENANCE MODE: LLM PROMPT DEBUG]");
		System.out.println("=".repeat(30));
		System.out.println(message);
		System.out.println("=".repeat(30) + "\n");
	}

	private String sendToLlm(String message) {
		// 実際のAPI通信ロジック（WebClient等）をここに実装
		System.out.println("[INFO] LLM API Call to: " + llmEndpoint);
		return "LLM Response Data (Placeholder)";
	}
}