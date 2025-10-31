import os
import json
import io
import base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import mysql.connector
import numpy as np
from PIL import Image
import face_recognition
from werkzeug.security import generate_password_hash, check_password_hash
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'))
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev_secret_key_change_me")

# MySQL config - set env vars in production
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'nmklop90',  # ← put your MySQL password here
    'database': 'facepay'
}


def get_db():
    conn = mysql.connector.connect(**DB_CONFIG)
    return conn

# Utility: convert base64 image data to numpy array
def b64_to_image(b64_data):
    header, encoded = b64_data.split(',', 1) if ',' in b64_data else (None, b64_data)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)

def compute_embedding_from_b64(b64_data):
    img_np = b64_to_image(b64_data)
    boxes = face_recognition.face_locations(img_np)
    if not boxes:
        return None, "No face detected"
    encodings = face_recognition.face_encodings(img_np, boxes)
    if not encodings:
        return None, "Failed to encode face"
    return encodings[0].tolist(), None

def find_user_by_embedding(embedding, threshold=0.6):
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, name, email, embedding_json, wallet_balance FROM users WHERE embedding_json IS NOT NULL")
    candidates = cur.fetchall()
    cur.close()
    conn.close()
    if not candidates:
        return None, None
    emb_np = np.array(embedding)
    best = None
    best_dist = float('inf')
    for c in candidates:
        try:
            stored = np.array(json.loads(c['embedding_json']))
            dist = np.linalg.norm(stored - emb_np)
            if dist < best_dist:
                best_dist = dist
                best = c
        except Exception as e:
            continue
    if best and best_dist <= threshold:
        return best, best_dist
    return None, best_dist

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        consent = request.form.get('consent')
        b64_image = request.form.get('image_data')
        if not consent:
            flash("You must consent to store your face embedding.")
            return redirect(url_for('register'))
        embedding, err = compute_embedding_from_b64(b64_image)
        if err:
            flash(err)
            return redirect(url_for('register'))
        # Insert user
        conn = get_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (name,email,embedding_json,wallet_balance) VALUES (%s,%s,%s,%s)",
                    (name, email, json.dumps(embedding), 10.0))  # give 10 coins starter
        conn.commit()
        cur.close()
        conn.close()
        flash("Registered successfully. You got 10 starter coins.")
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/login_face', methods=['POST'])
def login_face():
    b64_image = request.json.get('image_data')
    embedding, err = compute_embedding_from_b64(b64_image)
    if err:
        return jsonify({'success': False, 'error': err}), 400
    user, dist = find_user_by_embedding(embedding)
    if user:
        # create session
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        return jsonify({'success': True, 'user': {'id': user['id'], 'name': user['name'], 'balance': float(user['wallet_balance'])}})
    else:
        return jsonify({'success': False, 'error': 'No matching face found', 'distance': dist}), 404

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('dashboard.html', name=session.get('user_name'))

@app.route('/scan_send', methods=['GET','POST'])
def scan_send():
    # GET: show UI
    if request.method == 'GET':
        if 'user_id' not in session:
            return redirect(url_for('index'))
        return render_template('scan_send.html', name=session.get('user_name'))
    # POST: payload: image_data (recipient capture), amount, coin_id (optional)
    data = request.json
    b64_image = data.get('image_data')
    amount = float(data.get('amount', 0))
    sender_id = session.get('user_id')
    if amount <= 0:
        return jsonify({'success': False, 'error': 'Invalid amount'}), 400
    embedding, err = compute_embedding_from_b64(b64_image)
    if err:
        return jsonify({'success': False, 'error': err}), 400
    recipient, dist = find_user_by_embedding(embedding)
    if recipient is None:
        return jsonify({'success': False, 'error': 'Recipient not found'}), 404
    # debit sender, credit recipient
    conn = get_db()
    cur = conn.cursor()
    # get sender balance
    cur.execute("SELECT wallet_balance FROM users WHERE id=%s", (sender_id,))
    row = cur.fetchone()
    if not row:
        cur.close(); conn.close()
        return jsonify({'success': False, 'error': 'Sender not found'}), 404
    sender_balance = float(row[0])
    if sender_balance < amount:
        cur.close(); conn.close()
        return jsonify({'success': False, 'error': 'Insufficient balance'}), 400
    # update balances & insert transaction
    cur.execute("UPDATE users SET wallet_balance = wallet_balance - %s WHERE id = %s", (amount, sender_id))
    cur.execute("UPDATE users SET wallet_balance = wallet_balance + %s WHERE id = %s", (amount, recipient['id']))
    cur.execute("INSERT INTO transactions (sender_id, recipient_id, coin_id, amount) VALUES (%s,%s,%s,%s)",
                (sender_id, recipient['id'], None, amount))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({'success': True, 'recipient': {'id': recipient['id'], 'name': recipient['name']}, 'amount': amount})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Optional: simple endpoint to check if face registered
@app.route('/check_registered', methods=['POST'])
def check_registered():
    b64_image = request.json.get('image_data')
    embedding, err = compute_embedding_from_b64(b64_image)
    if err:
        return jsonify({'registered': False, 'error': err}), 400
    user, dist = find_user_by_embedding(embedding)
    if user:
        return jsonify({'registered': True, 'user': {'id': user['id'], 'name': user['name']}}, 200)
    else:
        return jsonify({'registered': False, 'distance': dist}), 200
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
