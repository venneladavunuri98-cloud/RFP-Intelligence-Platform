"""Local stand-in for a Slack Incoming Webhook (dry runs; nothing leaves the machine).

  python n8n/mock_slack.py [port] [logfile]    # default 8999, n8n/mock_slack_messages.jsonl
Each POSTed JSON body is appended to the log file and printed.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8999
LOG = sys.argv[2] if len(sys.argv) > 2 else "n8n/mock_slack_messages.jsonl"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            msg = json.loads(body)
        except ValueError:
            msg = {"raw": body.decode(errors="replace")}
        with open(LOG, "a") as f:
            f.write(json.dumps({"path": self.path, **msg}) + "\n")
        print("SLACK <-", msg.get("text", msg), flush=True)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
