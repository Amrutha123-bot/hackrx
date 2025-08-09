# from flask import Flask, request
# from dotenv import load_dotenv
# from pyngrok import ngrok
# import json
# import os
# ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN"))
# app = Flask(__name__)

# # This route matches your desired endpoint
# @app.route("/api/v1/hackrx/run", methods=["POST"])
# def github_webhook():
#     payload = request.json
#     print("\n📦 Received GitHub Webhook Payload:")
#     print(json.dumps(payload, indent=2))
#     return {"status": "Webhook received"}, 200

# if __name__ == "__main__":
#     if os.environ.get("HEROKU") == "1":
#         # Run normally on Heroku
#         app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
#     else:
#         # Run locally with ngrok
#         public_url = ngrok.connect(5000)
#         print(f"\n🚀 Ngrok Tunnel URL: {public_url}")
#         print("Use this in GitHub Webhook settings (append /api/v1/hackrx/run)")

#         app.run(port=5000)


# @app.route("/api/v1/hackrx/run", methods=["POST"])
# def github_webhook():
#     payload = request.json
#     with open("webhook_logs.json", "a") as f:
#         f.write(json.dumps(payload) + "\n")

#     print("\n📦 GitHub Webhook Payload:")
#     print(json.dumps(payload, indent=2))
#     return {"status": "Webhook received"}, 200

from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route("/api/v1/hackrx/run", methods=["POST"])
def github_webhook():
    event_type = request.headers.get("X-GitHub-Event")
    payload = request.json

    print(f"\n📩 Received GitHub event: {event_type}")
    print(json.dumps(payload, indent=2))

    return jsonify({"status": f"{event_type} event received ✅"}), 200

if __name__ == "__main__":
    app.run(port=5000)



# import os
# from flask import Flask, request
# import json

# app = Flask(__name__)

# @app.route("/api/v1/hackrx/run", methods=["POST"])
# def github_webhook():
#     payload = request.json
#     print("\n📦 GitHub Webhook Payload:")
#     print(json.dumps(payload, indent=2))
#     return {"status": "Webhook received"}, 200


# if __name__ == "__main__":
#     # Check if running on Heroku
#     is_heroku = os.environ.get("HEROKU") == "1"

#     if is_heroku:
#         # On Heroku: bind to the correct port
#         port = int(os.environ.get("PORT", 5000))
#         app.run(host="0.0.0.0", port=port)
#     else:
#         # Local dev with ngrok
#         try:
#             from pyngrok import ngrok, conf
#         except ImportError:
#             raise ImportError("You must install pyngrok to use ngrok in local development")

#         # Set ngrok path if needed
#         conf.get_default().ngrok_path = "/usr/local/bin/ngrok"

#         # Set auth token from env
#         authtoken = os.environ.get("NGROK_AUTHTOKEN")
#         if not authtoken:
#             print("❌ Please export NGROK_AUTHTOKEN before running.")
#             exit(1)

#         ngrok.set_auth_token(authtoken)

#         public_url = ngrok.connect(5000)
#         print(f"✅ ngrok tunnel running: {public_url}")
#         print(f"🔗 Your webhook endpoint is: {public_url}/api/v1/hackrx/run")

#         app.run(port=5000)
