import socket
import threading
import time
import json
from datetime import datetime

# Custom JSON parser (no lib, simple for industry level basic)
def parse_json(data):
    try:
        # Simple parse (for demo, real mein more robust bana sakte)
        data = data.strip()
        if data[0] == '{' and data[-1] == '}':
            pairs = data[1:-1].split(',')
            result = {}
            for pair in pairs:
                key, value = pair.split(':', 1)
                key = key.strip().strip('"')
                value = value.strip().strip('"')
                result[key] = value
            return result
        return {}
    except:
        return {}

# Custom rate limit (dict + time, 10/min per IP)
rate_limits = {}
RATE_LIMIT_CALLS = 10
RATE_LIMIT_PERIOD = 60

def check_rate_limit(ip):
    now = time.time()
    if ip not in rate_limits:
        rate_limits[ip] = []
    rate_limits[ip] = [t for t in rate_limits[ip] if now - t < RATE_LIMIT_PERIOD]
    if len(rate_limits[ip]) >= RATE_LIMIT_CALLS:
        return False
    rate_limits[ip].append(now)
    return True

# Dummy calls (baaki layers ke baad real banaenge)
def verify_auth(header):
    # Dummy auth check (real in Auth Layer)
    return {"subscription_tier": "ultra"} if "Bearer dummy_token" in header else None

def verify_subscription(user):
    # Dummy billing check (real in Billing Layer)
    return {"max_tokens": 8000, "allow_tools": True} if user else None

def call_orchestrator(question, config, conversation_id):
    # Dummy orchestrator call (real in Orchestrator Layer)
    return "Response from AGI: Hello from custom API!"

# Custom HTTP response builder
def build_response(status, body):
    response = f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}"
    return response.encode()

# Handle request
def handle_client(conn, addr):
    data = conn.recv(4096).decode('utf-8')
    if not data:
        conn.close()
        return

    lines = data.split('\r\n')
    request_line = lines[0]
    method, path, _ = request_line.split()

    # Headers parse
    headers = {}
    for line in lines[1:]:
        if line == '':
            break
        key, value = line.split(':', 1)
        headers[key.strip()] = value.strip()

    # Body parse (POST)
    body = lines[-1] if method == 'POST' else ''

    # Rate limit check
    if not check_rate_limit(addr[0]):
        conn.send(build_response("429 Too Many Requests", json.dumps({"error": "Rate limit exceeded"})))
        conn.close()
        return

    if method == 'POST' and path == '/chat':
        request = parse_json(body)
        if "message" not in request or "conversation_id" not in request:
            conn.send(build_response("400 Bad Request", json.dumps({"error": "Invalid request"})))
            conn.close()
            return

        # Auth check
        auth_header = headers.get("Authorization", "")
        user = verify_auth(auth_header)
        if not user:
            conn.send(build_response("401 Unauthorized", json.dumps({"error": "Invalid token"})))
            conn.close()
            return

        # Subscription verify
        config = verify_subscription(user)
        if not config:
            conn.send(build_response("403 Forbidden", json.dumps({"error": "Subscription invalid"})))
            conn.close()
            return

        # Orchestrator call
        response = call_orchestrator(request["message"], config, request["conversation_id"])
        conn.send(build_response("200 OK", json.dumps({"response": response})))
    else:
        conn.send(build_response("404 Not Found", json.dumps({"error": "Not found"})))

    conn.close()

# Custom server
def start_server(host='0.0.0.0', port=8000):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)  # Backlog for scale
    print(f"Custom API server running on {host}:{port}")

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()

if __name__ == "__main__":
    start_server()