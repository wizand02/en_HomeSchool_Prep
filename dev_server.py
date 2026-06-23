import http.server
import socketserver
import json
import os
import urllib.parse

PORT = 8000
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

class IDERequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            return

        elif path == "/api/files":
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()
            
            files_list = []
            for root, dirs, files in os.walk(WORKSPACE_DIR):
                # .git 이나 output_sounds 폴더는 리스트에서 간소화
                if '.git' in dirs:
                    dirs.remove('.git')
                
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), WORKSPACE_DIR)
                    # 제외할 확장자나 임시 파일 필터링
                    if not rel_path.startswith("output_sounds") and not rel_path.endswith(".pyc"):
                        files_list.append({
                            "name": file,
                            "path": rel_path.replace("\\", "/"),
                            "size": os.path.getsize(os.path.join(root, file))
                        })
            
            self.wfile.write(json.dumps(files_list).encode('utf-8'))
            return

        elif path == "/api/file":
            query = urllib.parse.parse_qs(parsed_path.query)
            file_rel_path = query.get("path", [None])[0]
            if not file_rel_path:
                self.send_error(400, "Missing path parameter")
                return

            full_path = os.path.abspath(os.path.join(WORKSPACE_DIR, file_rel_path))
            # 보안용: 워크스페이스 디렉토리 외부의 파일 접근 차단
            if not full_path.startswith(WORKSPACE_DIR):
                self.send_error(403, "Access Denied")
                return

            if not os.path.exists(full_path) or os.path.isdir(full_path):
                self.send_error(404, "File Not Found")
                return

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            except Exception as e:
                self.send_error(500, str(e))
            return

        else:
            super().do_GET()

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == "/api/save":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                file_rel_path = data.get("path")
                content = data.get("content")

                if not file_rel_path:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Missing path")
                    return

                full_path = os.path.abspath(os.path.join(WORKSPACE_DIR, file_rel_path))
                if not full_path.startswith(WORKSPACE_DIR):
                    self.send_response(403)
                    self.end_headers()
                    self.wfile.write(b"Access Denied")
                    return

                # 백업본 생성 후 저장
                if os.path.exists(full_path):
                    backup_path = full_path + ".bak"
                    with open(full_path, "r", encoding="utf-8") as src:
                        backup_content = src.read()
                    with open(backup_path, "w", encoding="utf-8") as bak:
                        bak.write(backup_content)

                with open(full_path, "w", encoding="utf-8") as dest:
                    dest.write(content)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "backup": True}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
            return

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local Web IDE - en_voca_sounds</title>
    <!-- Tailwind CSS (Aesthetics) -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <!-- Monaco Editor (Premium VS Code Editor) -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #0f172a;
            color: #e2e8f0;
        }
        .code-font {
            font-family: 'Fira Code', monospace;
        }
    </style>
</head>
<body class="h-screen flex flex-col overflow-hidden">

    <!-- Header -->
    <header class="bg-slate-900 border-b border-slate-800 px-6 py-4 flex items-center justify-between shadow-lg">
        <div class="flex items-center space-x-3">
            <span class="text-2xl">⚡</span>
            <h1 class="text-lg font-semibold tracking-wider text-slate-100">LOCAL DEVELOPMENT WEB IDE</h1>
            <span class="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/30 font-mono">Offline-Safe</span>
        </div>
        <div id="status-message" class="text-sm text-slate-400 font-mono">대기 중...</div>
    </header>

    <!-- Main Workspace Layout -->
    <div class="flex-1 flex overflow-hidden">
        
        <!-- Sidebar: File Explorer -->
        <aside class="w-80 bg-slate-900/60 backdrop-blur border-r border-slate-800 flex flex-col">
            <div class="p-4 border-b border-slate-800 flex items-center justify-between">
                <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">📁 Project Explorer</span>
                <button onclick="loadFiles()" class="text-xs text-indigo-400 hover:text-indigo-300 transition">🔄 새로고침</button>
            </div>
            <div id="file-list" class="flex-1 overflow-y-auto p-2 space-y-1">
                <!-- Files will load here -->
                <div class="text-slate-500 p-4 text-xs">로딩 중...</div>
            </div>
        </aside>

        <!-- Editor Container -->
        <main class="flex-1 flex flex-col bg-[#1e1e1e]">
            <!-- Editor Tabs / Toolbar -->
            <div class="bg-slate-900 border-b border-slate-800 px-4 py-2 flex items-center justify-between">
                <div class="flex items-center space-x-2">
                    <span class="text-sm font-mono text-indigo-400 font-semibold" id="active-filename">파일을 선택해 주세요.</span>
                    <span class="text-xs text-slate-500 hidden" id="backup-badge">(자동 백업 활성)</span>
                </div>
                <button onclick="saveActiveFile()" id="save-btn" class="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-1.5 rounded text-sm font-medium transition shadow-md flex items-center space-x-1.5 opacity-50 cursor-not-allowed" disabled>
                    <span>💾</span> <span>저장하기 (Save)</span>
                </button>
            </div>
            
            <!-- Monaco Editor Container -->
            <div id="editor-container" class="flex-1 w-full"></div>
        </main>
    </div>

    <!-- Footer / Console Guidance -->
    <footer class="bg-slate-900 border-t border-slate-800 px-6 py-3 text-xs text-slate-400 flex items-center justify-between">
        <div>
            <span>📍 workspace: </span><span class="text-slate-200 font-mono">c:\\Users\\wizan\\dev_gina\\en_voca_sounds</span>
        </div>
        <div>
            <span>💡 Streamlit 구동 명령어: </span><span class="bg-slate-800 text-indigo-300 px-2 py-0.5 rounded font-mono select-all">streamlit run app.py</span>
        </div>
    </footer>

    <!-- Script to Init Monaco and handle API -->
    <script>
        let editor = null;
        let activeFilePath = null;

        // Initialize Monaco Editor
        require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.39.0/min/vs' }});
        require(['vs/editor/editor.main'], function() {
            editor = monaco.editor.create(document.getElementById('editor-container'), {
                value: "# 파일을 왼쪽 탐색기에서 선택하시면 코드가 로드됩니다.\\n# 수정 후 상단의 [저장하기]를 누르면 로컬 파일에 즉시 저장됩니다.\\n",
                language: 'python',
                theme: 'vs-dark',
                automaticLayout: true,
                fontSize: 14,
                fontFamily: 'Fira Code, monospace',
                tabSize: 4,
                minimap: { enabled: true }
            });
            
            // Listen to content change to enable save button
            editor.onDidChangeModelContent(() => {
                if (activeFilePath) {
                    enableSaveButton(true);
                }
            });
        });

        // Load files list
        async function loadFiles() {
            const status = document.getElementById('status-message');
            status.innerText = "파일 탐색 중...";
            try {
                const res = await fetch('/api/files');
                const files = await res.json();
                const listContainer = document.getElementById('file-list');
                listContainer.innerHTML = '';

                if (files.length === 0) {
                    listContainer.innerHTML = '<div class="text-slate-500 p-4 text-xs">파일이 없습니다.</div>';
                    return;
                }

                files.forEach(file => {
                    const button = document.createElement('button');
                    button.className = "w-full text-left px-3 py-2 rounded text-sm text-slate-300 hover:bg-slate-800 hover:text-white transition flex items-center justify-between group";
                    button.onclick = () => loadFile(file.path);
                    
                    let icon = "📄";
                    if (file.name.endsWith(".py")) icon = "🐍";
                    if (file.name.endsWith(".txt") || file.name.endsWith(".TXT")) icon = "📝";
                    if (file.name.endsWith(".xlsx")) icon = "📊";

                    button.innerHTML = `
                        <div class="flex items-center space-x-2 truncate">
                            <span>${icon}</span>
                            <span class="truncate font-mono">${file.path}</span>
                        </div>
                        <span class="text-[10px] text-slate-500 group-hover:text-slate-400 font-mono">${(file.size / 1024).toFixed(1)} KB</span>
                    `;
                    listContainer.appendChild(button);
                });
                status.innerText = "대기 중";
            } catch (err) {
                console.error(err);
                status.innerText = "파일 목록 로드 실패";
            }
        }

        // Load specific file content into Monaco Editor
        async function loadFile(path) {
            const status = document.getElementById('status-message');
            status.innerText = `${path} 로딩 중...`;
            
            // xlsx 파일 등은 텍스트 편집기 로드 불가 안내
            if (path.endsWith(".xlsx")) {
                alert("엑셀 파일(.xlsx)은 텍스트 에디터에서 직접 수정할 수 없습니다.");
                status.innerText = "대기 중";
                return;
            }

            try {
                const res = await fetch(`/api/file?path=${encodeURIComponent(path)}`);
                if (!res.ok) throw new Error("로드 실패");
                const content = await res.text();
                
                activeFilePath = path;
                document.getElementById('active-filename').innerText = path;
                document.getElementById('backup-badge').classList.remove('hidden');

                // Determine language
                let lang = 'plaintext';
                if (path.endsWith('.py')) lang = 'python';
                if (path.endsWith('.json')) lang = 'json';
                if (path.endsWith('.html')) lang = 'html';
                
                const model = monaco.editor.createModel(content, lang);
                editor.setModel(model);

                // Listen to edits on new model
                editor.onDidChangeModelContent(() => {
                    enableSaveButton(true);
                });

                enableSaveButton(false);
                status.innerText = `${path} 로드 완료`;
            } catch (err) {
                console.error(err);
                status.innerText = "파일 로드 오류";
            }
        }

        // Save active file
        async function saveActiveFile() {
            if (!activeFilePath) return;
            const status = document.getElementById('status-message');
            status.innerText = "파일 저장 중...";
            const content = editor.getValue();

            try {
                const res = await fetch('/api/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: activeFilePath, content: content })
                });

                if (res.ok) {
                    status.innerText = `${activeFilePath} 저장 및 백업 완료!`;
                    enableSaveButton(false);
                    // 1초 뒤 대기 중으로 복귀
                    setTimeout(() => {
                        if (status.innerText.includes("저장 및 백업 완료")) {
                            status.innerText = "대기 중";
                        }
                    }, 2000);
                } else {
                    throw new Error("저장 오류");
                }
            } catch (err) {
                console.error(err);
                status.innerText = "저장 실패";
                alert("파일 저장에 실패했습니다. 서버 콘솔을 확인해 주세요.");
            }
        }

        function enableSaveButton(enable) {
            const btn = document.getElementById('save-btn');
            if (enable) {
                btn.removeAttribute('disabled');
                btn.classList.remove('opacity-50', 'cursor-not-allowed');
                btn.classList.add('hover:bg-indigo-500');
            } else {
                btn.setAttribute('disabled', 'true');
                btn.classList.add('opacity-50', 'cursor-not-allowed');
                btn.classList.remove('hover:bg-indigo-500');
            }
        }

        // Initial Load
        window.onload = loadFiles;
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    handler = IDERequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"=========================================================")
        print(f"  Local Web IDE Server Started!")
        print(f"  👉 Open browser: http://localhost:{PORT}")
        print(f"  👉 WORKSPACE: {WORKSPACE_DIR}")
        print(f"  * Note: Modifying code creates a '.bak' copy automatically.")
        print(f"=========================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
