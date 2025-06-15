import json
import subprocess
import sys
import os
from typing import Dict, Any, List, Optional
import google.generativeai as genai


class MCPClient:
    """簡單的 MCP 客戶端，透過 stdio 與 MCP 伺服器通訊"""
    
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.process = None
        
    def __enter__(self):
        """啟動伺服器程序"""
        self.process = subprocess.Popen(
            [sys.executable, self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        
        # 讀取伺服器啟動訊息
        ready_msg = self.process.stdout.readline()
        print(f"🚀 {ready_msg.strip()}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """關閉伺服器程序"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            
    def call_tool(self, tool: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """呼叫 MCP 工具"""
        if not self.process:
            raise RuntimeError("客戶端未啟動")
            
        request = {
            "tool": tool,
            "args": args or {}
        }
        
        # 發送請求
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json)
        self.process.stdin.flush()
        
        # 讀取回應
        response_line = self.process.stdout.readline()
        response = json.loads(response_line)
        
        return response


class GeminiMCPAgent:
    """使用 Gemini 的智能 MCP 代理"""
    
    def __init__(self, api_key: str, mcp_client: MCPClient):
        genai.configure(api_key=api_key)
        
        # 嘗試不同的模型名稱，優先使用最新的 gemini-2.0-flash
        model_names = [
            'gemini-2.0-flash',
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-pro',
            'models/gemini-2.0-flash',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro'
        ]
        
        self.model = None
        for model_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name)
                print(f"✅ 使用模型: {model_name}")
                break
            except Exception as e:
                print(f"⚠️  嘗試模型 {model_name} 失敗: {str(e)}")
                continue
        
        if not self.model:
            # 列出可用模型
            try:
                available_models = list(genai.list_models())
                print("📋 可用的模型:")
                for model in available_models:
                    if 'generateContent' in model.supported_generation_methods:
                        print(f"   - {model.name}")
                
                # 使用第一個支援 generateContent 的模型
                for model in available_models:
                    if 'generateContent' in model.supported_generation_methods:
                        self.model = genai.GenerativeModel(model.name)
                        print(f"✅ 自動選擇模型: {model.name}")
                        break
            except Exception as e:
                print(f"❌ 無法列出可用模型: {str(e)}")
        
        if not self.model:
            raise RuntimeError("無法找到可用的 Gemini 模型")
        
        self.mcp_client = mcp_client
        self.conversation_history = []
        
        # 可用的工具描述
        self.available_tools = {
            "test": {
                "description": "測試伺服器是否正常運行",
                "parameters": {}
            },
            "search": {
                "description": "在詐欺資料集中搜尋相關文件",
                "parameters": {
                    "query": "搜尋查詢字串",
                    "top_k": "返回的結果數量 (預設: 5)"
                }
            },
            "expand_search": {
                "description": "使用 Gemini 擴充查詢後再搜尋",
                "parameters": {
                    "query": "搜尋查詢字串",
                    "top_k": "返回的結果數量 (預設: 5)"
                }
            },
            "read_fraud_data": {
                "description": "讀取完整的詐欺判決摘要資料集",
                "parameters": {}
            },
            "evaluate_fraud": {
                "description": "評估 BM25 在詐欺查詢上的效能",
                "parameters": {
                    "top_k": "評估時考慮的前 k 個結果 (預設: 10)"
                }
            }
        }
    
    def _create_system_prompt(self) -> str:
        """建立系統提示"""
        tools_desc = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.available_tools.items()
        ])

        return f"""你是一個智能助手，可以幫助使用者查詢和分析詐欺相關資料。

你有以下可用的工具：
{tools_desc}

通常請先使用 `search` 工具檢索，觀察結果後如有需要再使用
`expand_search` 透過 Gemini 擴充查詢後重新搜尋。

請根據使用者的問題，判斷是否需要使用這些工具。如果需要使用工具，請以 JSON 格式回應：
{{
    "action": "use_tool",
    "tool": "工具名稱",
    "args": {{"參數名": "參數值"}},
    "reasoning": "為什麼要使用這個工具的原因"
}}

如果不需要使用工具，請直接回答使用者的問題：
{{
    "action": "respond",
    "response": "你的回答"
}}

請用繁體中文回應。"""

    def _analyze_user_input(self, user_input: str) -> Dict[str, Any]:
        """分析使用者輸入並決定是否需要使用工具"""
        
        # 建立對話歷史
        history_text = "\n".join([
            f"使用者: {entry['user']}\n助手: {entry['assistant']}"
            for entry in self.conversation_history[-3:]  # 只保留最近3輪對話
        ])
        
        prompt = f"""
{self._create_system_prompt()}

對話歷史：
{history_text}

使用者最新問題：{user_input}

請分析這個問題並決定下一步行動。
"""
        
        try:
            # 使用更安全的生成配置
            generation_config = genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # 嘗試解析 JSON 回應
            response_text = response.text.strip()
            
            # 移除可能的 markdown 代碼塊標記
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            return json.loads(response_text)
            
        except json.JSONDecodeError:
            # 如果無法解析 JSON，假設是直接回應
            return {
                "action": "respond",
                "response": response.text if hasattr(response, 'text') else str(response)
            }
        except Exception as e:
            return {
                "action": "respond",
                "response": f"抱歉，我在處理您的請求時遇到了問題：{str(e)}"
            }
    
    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """執行 MCP 工具並格式化結果"""
        try:
            response = self.mcp_client.call_tool(tool_name, args)
            
            if "error" in response:
                return f"❌ 工具執行錯誤：{response['error']}"
            
            result = response["result"]
            
            # 根據不同工具格式化輸出
            if tool_name == "test":
                return f"✅ {result}"
            
            elif tool_name == "search":
                if not result:
                    return "🔍 沒有找到相關結果"

                formatted_results = ["🔍 搜尋結果："]
                for i, item in enumerate(result[:5], 1):
                    formatted_results.append(
                        f"\n{i}. 文件ID: {item['doc_id']}"
                        f"\n   相關度: {item['score']:.4f}"
                        f"\n   內容: {item['text'][:200]}{'...' if len(item['text']) > 200 else ''}\n"
                    )
                return "\n".join(formatted_results)

            elif tool_name == "expand_search":
                results = result.get("results", [])
                expanded_query = result.get("expanded_query", "")
                if not results:
                    return f"🔍 擴充後查詢：{expanded_query}\n沒有找到相關結果"

                formatted_results = [f"🔍 擴充後查詢：{expanded_query}"]
                for i, item in enumerate(results[:5], 1):
                    formatted_results.append(
                        f"\n{i}. 文件ID: {item['doc_id']}"
                        f"\n   相關度: {item['score']:.4f}"
                        f"\n   內容: {item['text'][:200]}{'...' if len(item['text']) > 200 else ''}\n"
                    )
                return "\n".join(formatted_results)
            
            elif tool_name == "read_fraud_data":
                return f"📊 詐欺資料集包含 {len(result)} 筆記錄"
            
            elif tool_name == "evaluate_fraud":
                return (f"📈 評估結果：\n"
                       f"   準確率: {result['accuracy']:.4f}\n"
                       f"   MRR: {result['mrr']:.4f}")
            
            else:
                return f"📋 結果：{json.dumps(result, ensure_ascii=False, indent=2)}"
                
        except Exception as e:
            return f"❌ 執行工具時發生錯誤：{str(e)}"
    
    def chat(self, user_input: str) -> str:
        """處理使用者輸入並回應"""
        print(f"🤔 分析中...")
        
        # 分析使用者輸入
        analysis = self._analyze_user_input(user_input)
        
        if analysis["action"] == "use_tool":
            tool_name = analysis["tool"]
            args = analysis.get("args", {})
            reasoning = analysis.get("reasoning", "")
            
            print(f"🔧 準備使用工具: {tool_name}")
            if reasoning:
                print(f"💭 原因: {reasoning}")
            
            # 執行工具
            tool_result = self._execute_tool(tool_name, args)
            
            # 讓 Gemini 解釋結果
            explain_prompt = f"""
根據以下工具執行結果，請用自然語言向使用者解釋：

使用者問題：{user_input}
工具結果：{tool_result}

請提供清楚、有用的解釋。
"""
            
            try:
                explanation = self.model.generate_content(explain_prompt)
                final_response = f"{tool_result}\n\n💡 {explanation.text}"
            except:
                final_response = tool_result
            
        else:
            final_response = analysis["response"]
        
        # 記錄對話歷史
        self.conversation_history.append({
            "user": user_input,
            "assistant": final_response
        })
        
        return final_response


def main():
    """主程式"""
    # 檢查 API 金鑰
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ 請設定 GEMINI_API_KEY 環境變數")
        print("   export GEMINI_API_KEY=your_api_key_here")
        return
    
    print("🌟 Gemini MCP 智能助手啟動中...")
    
    try:
        with MCPClient("mcp_server.py") as mcp_client:
            agent = GeminiMCPAgent(api_key, mcp_client)
            
            print("\n✨ 助手已準備就緒！您可以詢問關於詐欺資料的任何問題。")
            print("💡 例如：")
            print("   - 搜尋 money laundering 相關文件")
            print("   - 詐欺資料集有多少筆資料？")
            print("   - 評估一下搜尋系統的效能")
            print("   - 輸入 'quit' 或 'exit' 結束\n")
            
            while True:
                try:
                    user_input = input("👤 您: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', '退出', '結束']:
                        print("👋 再見！")
                        break
                    
                    if not user_input:
                        continue
                    
                    response = agent.chat(user_input)
                    print(f"\n🤖 助手: {response}\n")
                    
                except KeyboardInterrupt:
                    print("\n👋 再見！")
                    break
                except Exception as e:
                    print(f"❌ 發生錯誤：{str(e)}")
                    
    except Exception as e:
        print(f"❌ 無法啟動 MCP 伺服器：{str(e)}")
        print("請確認 mcp_server.py 檔案存在且相關依賴已安裝")


if __name__ == "__main__":
    main()