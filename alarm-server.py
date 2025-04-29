from flask import Flask, request, jsonify
import threading
import playsound

app = Flask(__name__)

# Alarm function
def play_alarm():
    try:
        playsound.playsound("alarm.mp3")
    except Exception as e:
        print(f"Error playing alarm: {e}")

@app.route("/alert", methods=["POST"])
def alert():
    data = request.get_json()
    if data.get("status") == "fire":
        print("🚨 Fire detected! Playing alarm...")
        threading.Thread(target=play_alarm).start()  # Play alarm sound asynchronously
        return jsonify({"message": "Alarm triggered!"}), 200
    else:
        return jsonify({"message": "No action taken"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
