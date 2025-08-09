from flask import Flask, request

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def github_webhook():
    data = request.json
    print("Webhook received:", data)
    return '', 200

app.run(port=5000)