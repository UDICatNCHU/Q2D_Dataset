"""Flask server providing a simple chat interface to the Gemini agent."""

import os
from flask import Flask, request, jsonify, render_template

from gemini_mcp_client import GeminiMCPAgent
from mcp_client import MCPClient

app = Flask(__name__)

# Initialize Gemini agent and MCP server
_API_KEY = os.getenv("GEMINI_API_KEY")
if not _API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required")

_MCP_CLIENT = MCPClient("mcp_server.py")
_MCP_CLIENT.start()
_AGENT = GeminiMCPAgent(_API_KEY, _MCP_CLIENT)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(force=True)
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Missing message'}), 400
    try:
        response = _AGENT.chat(message)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify({'response': response})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8000)
    finally:
        _MCP_CLIENT.stop()
