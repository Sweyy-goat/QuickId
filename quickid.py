# quickid.py — Inline Tailwind FacePay (MySQL) single-file app
# Requirements: Flask, mysql-connector-python, numpy, Pillow, face_recognition
# Usage: python quickid.py

import os, io, json, base64
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for
import mysql.connector
import numpy as np
from PIL import Image
import face_recognition

# ---------- CONFIG ----------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev_secret_change_me")
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASS", "nmklop90"),  # change to your MySQL password
    "database": os.environ.get("DB_NAME", "facepay"),
    "port": int(os.environ.get("DB_PORT", 3306))
}
# ----------------------------

def get_db():
    return mysql.connector.connect(**DB_CONFIG)

# ---------- Ensure schema (creates users, coins, transactions) ----------
def ensure_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("CREATE DATABASE IF NOT EXISTS `%s`" % DB_CONFIG["database"])
    conn.database = DB_CONFIG["database"]

    # users: stores embeddings as JSON and wallet balance
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
      id INT AUTO_INCREMENT PRIMARY KEY,
      name VARCHAR(150),
      email VARCHAR(200) UNIQUE,
      embedding_json LONGTEXT,
      wallet_balance DOUBLE DEFAULT 0,
      last_transaction DATETIME NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)

    # coins: optional metadata for coins (kept minimal)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS coins (
      id INT AUTO_INCREMENT PRIMARY KEY,
      name VARCHAR(100),
      symbol VARCHAR(20),
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)

    # transactions: logs transfers
    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
      id INT AUTO_INCREMENT PRIMARY KEY,
      sender_id INT,
      recipient_id INT,
      amount DOUBLE,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (sender_id) REFERENCES users(id) ON DELETE SET NULL,
      FOREIGN KEY (recipient_id) REFERENCES users(id) ON DELETE SET NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    conn.commit()
    cur.close(); conn.close()

ensure_db()

# ---------- Utilities ----------
def b64_to_image_array(b64_data):
    header, encoded = (b64_data.split(",", 1) if "," in b64_data else (None, b64_data))
    try:
        img_bytes = base64.b64decode(encoded)
    except Exception as e:
        raise ValueError(f"Bad base64 data: {e}")
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)

def compute_embedding_from_b64(b64_data, resize_width=800):
    try:
        img_np = b64_to_image_array(b64_data)
    except Exception as e:
        return None, f"Bad image data: {e}"
    # downscale for speed
    h, w = img_np.shape[:2]
    if w > resize_width:
        scale = resize_width / w
        img_pil = Image.fromarray(img_np)
        img_pil = img_pil.resize((resize_width, int(h*scale)))
        img_np = np.array(img_pil)
    boxes = face_recognition.face_locations(img_np)
    if not boxes:
        return None, "No face detected"
    encs = face_recognition.face_encodings(img_np, boxes)
    if not encs:
        return None, "Failed to encode face"
    return encs[0].tolist(), None

def find_user_by_embedding(embedding, threshold=0.6):
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, name, email, embedding_json, wallet_balance FROM users WHERE embedding_json IS NOT NULL")
    rows = cur.fetchall()
    cur.close(); conn.close()
    if not rows:
        return None, None
    emb_np = np.array(embedding)
    best = None
    best_dist = float("inf")
    for r in rows:
        try:
            stored = np.array(json.loads(r["embedding_json"]))
            dist = np.linalg.norm(stored - emb_np)
            if dist < best_dist:
                best_dist = dist; best = r
        except Exception:
            continue
    if best and best_dist <= threshold:
        return best, best_dist
    return None, best_dist

# ---------- Base HTML (uses Jinja, content safe) ----------
BASE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>FacePay</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 min-h-screen">
  <nav class="bg-white shadow">
    <div class="max-w-4xl mx-auto px-4 py-3 flex justify-between items-center">
      <div class="text-lg font-semibold text-sky-600">FacePay</div>
      <div>
        {% if session.get('user_name') %}
          <span class="mr-4 text-slate-700">Hi, {{ session.get('user_name') }}</span>
          <a href="/dashboard" class="mr-3 text-slate-700">Dashboard</a>
          <a href="/logout" class="text-red-500">Logout</a>
        {% else %}
          <a href="/" class="mr-3 text-slate-700">Home</a>
          <a href="/register" class="text-slate-700">Register</a>
        {% endif %}
      </div>
    </div>
  </nav>
  <main class="max-w-4xl mx-auto p-6">
    {{ content|safe }}
  </main>
</body>
</html>
"""

from flask import render_template_string

# ---------- Routes ----------
@app.route("/")
def index():
    content = """
    <div class="bg-white rounded-xl shadow p-6">
      <h2 class="text-2xl font-bold text-slate-800">Welcome to FacePay</h2>
      <p class="text-slate-600 mt-2">Use your face to login or register. Send coins to friends securely.</p>
      <div class="mt-4">
        <button onclick="location.href='/register'" class="px-4 py-2 bg-sky-600 text-white rounded mr-2">Register</button>
        <button onclick="startLogin()" class="px-4 py-2 bg-slate-700 text-white rounded">Login with Face</button>
      </div>
      <div id="msg" class="mt-4 text-sm text-red-500"></div>
      <div id="preview" class="mt-4"></div>
    </div>

<script>
async function startLogin(){
  const msg = document.getElementById('msg'); msg.textContent='Requesting camera...';
  try{
    const video = document.createElement('video'); video.className='rounded shadow'; video.style.maxWidth='320px';
    document.getElementById('preview').innerHTML=''; document.getElementById('preview').appendChild(video);
    const stream = await navigator.mediaDevices.getUserMedia({video:true});
    video.srcObject = stream; await video.play();
    await new Promise(r=>setTimeout(r,700));
    const canvas = document.createElement('canvas'); canvas.width=video.videoWidth||320; canvas.height=video.videoHeight||240;
    const ctx = canvas.getContext('2d'); ctx.drawImage(video,0,0,canvas.width,canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg');
    stream.getTracks().forEach(t=>t.stop());
    msg.textContent='Logging in...';
    const res = await fetch('/login_face',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_data:dataUrl})});
    const text = await res.text(); let j;
    try{ j = JSON.parse(text); } catch(e){ msg.textContent='Server error: '+text; return; }
    if(j.success){ window.location='/dashboard'; } else { msg.textContent = j.error || 'Login failed'; }
  } catch(e){ msg.textContent = 'Camera error: '+e.message; }
}
</script>
"""
    return render_template_string(BASE_TEMPLATE, content=content)

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "GET":
        content = """
        <div class="bg-white rounded-xl shadow p-6">
          <h2 class="text-2xl font-bold">Register — Capture Face</h2>
          <p class="text-slate-600 mt-2">Provide name & email, consent, then take a photo. You receive starter coins.</p>
          <div class="mt-4">
            <input id="name" class="border p-2 rounded mr-2" placeholder="Full name" />
            <input id="email" class="border p-2 rounded" placeholder="Email" />
          </div>
          <div class="mt-3">
            <label class="text-sm"><input id="consent" type="checkbox" class="mr-2" /> I consent to storing my face embedding</label>
          </div>
          <div class="mt-4">
            <video id="reg_video" autoplay class="rounded shadow"></video>
            <div class="mt-3">
              <button onclick="captureAndRegister()" class="px-4 py-2 bg-sky-600 text-white rounded">Capture & Register</button>
            </div>
            <div id="rmsg" class="mt-3 text-sm text-red-500"></div>
          </div>
        </div>

<script>
async function ensureCamera(el){
  if(el.srcObject) return;
  try{ const stream = await navigator.mediaDevices.getUserMedia({video:true}); el.srcObject = stream; await el.play(); } catch(e){ throw e; }
}
(async ()=>{ const v=document.getElementById('reg_video'); try{ await ensureCamera(v); } catch(e){ document.getElementById('rmsg').textContent='Camera error: '+e.message; } })();

async function captureAndRegister(){
  const msg=document.getElementById('rmsg'); const name=document.getElementById('name').value.trim();
  const email=document.getElementById('email').value.trim(); const consent=document.getElementById('consent').checked;
  if(!name||!email){ msg.textContent='Name and email required'; return; }
  if(!consent){ msg.textContent='Consent required'; return; }
  msg.textContent='Capturing...';
  try{
    const v=document.getElementById('reg_video'); const canvas=document.createElement('canvas');
    canvas.width=v.videoWidth||320; canvas.height=v.videoHeight||240;
    const ctx=canvas.getContext('2d'); ctx.drawImage(v,0,0,canvas.width,canvas.height);
    const dataUrl=canvas.toDataURL('image/jpeg'); msg.textContent='Uploading...';
    const res=await fetch('/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name,email:email,image_data:dataUrl})});
    const text=await res.text(); try{ const j=JSON.parse(text); if(j.success){ msg.textContent=j.message; setTimeout(()=>window.location='/',900); } else { msg.textContent='Error: '+(j.message||'unknown'); } }catch(e){ msg.textContent='Server error: '+text; }
  }catch(e){ msg.textContent='Camera error: '+e.message; }
}
</script>
"""
        return render_template_string(BASE_TEMPLATE, content=content)

    # POST
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "message": "Invalid JSON"}), 400
    name = (data or {}).get("name"); email = (data or {}).get("email"); b64 = (data or {}).get("image_data")
    if not (name and email and b64):
        return jsonify({"success": False, "message": "Missing name/email/image"}), 400
    emb, err = compute_embedding_from_b64(b64)
    if err:
        return jsonify({"success": False, "message": err}), 400

    # prevent duplicate face registration (tight threshold)
    existing, dist = find_user_by_embedding(emb, threshold=0.5)
    if existing:
        return jsonify({"success": False, "message": f"Face already registered as {existing['name']} (dist={dist:.3f})"}), 400

    emb_json = json.dumps(emb)
    conn = get_db(); cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (name,email,embedding_json,wallet_balance,last_transaction) VALUES (%s,%s,%s,%s,%s)",
                    (name, email, emb_json, 100.0, datetime.utcnow()))
        conn.commit()
        return jsonify({"success": True, "message": f"Registered {name}. You got 100 coins."})
    except mysql.connector.IntegrityError:
        conn.rollback(); return jsonify({"success": False, "message": "Email already registered"}), 400
    except Exception as e:
        conn.rollback(); return jsonify({"success": False, "message": f"DB error: {e}"}), 500
    finally:
        cur.close(); conn.close()

@app.route("/login_face", methods=["POST"])
def login_face():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON"}), 400
    b64 = (data or {}).get("image_data")
    if not b64:
        return jsonify({"success": False, "error": "Missing image_data"}), 400
    emb, err = compute_embedding_from_b64(b64)
    if err:
        return jsonify({"success": False, "error": err}), 400
    user, dist = find_user_by_embedding(emb)
    if user:
        session["user_id"] = user["id"]; session["user_name"] = user["name"]
        return jsonify({"success": True, "user": {"id": user["id"], "name": user["name"], "balance": float(user["wallet_balance"] or 0)}})
    return jsonify({"success": False, "error": "No matching face", "distance": dist}), 404

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/")
    uid = session["user_id"]
    conn = get_db(); cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id,name,email,wallet_balance,last_transaction FROM users WHERE id=%s", (uid,))
    user = cur.fetchone()
    cur.execute("""SELECT t.id,t.sender_id,t.recipient_id,t.amount,t.created_at,
                   u.name AS sender_name, ur.name AS recipient_name
                   FROM transactions t
                   LEFT JOIN users u ON u.id=t.sender_id
                   LEFT JOIN users ur ON ur.id=t.recipient_id
                   WHERE t.sender_id=%s OR t.recipient_id=%s
                   ORDER BY t.created_at DESC LIMIT 20""", (uid, uid))
    txs = cur.fetchall()
    cur.close(); conn.close()

    # pass txs to Jinja directly (avoid building HTML in python)
    content = render_template_string("""
    <div class="bg-white rounded-xl shadow p-6">
      <div class="flex justify-between items-center">
        <div>
          <h2 class="text-2xl font-bold">Dashboard</h2>
          <p class="text-slate-600">Hello, {{ user.name }}</p>
        </div>
        <div class="text-right">
          <div class="text-sm text-slate-500">Balance</div>
          <div id="balance" class="text-2xl font-semibold text-sky-600">{{ "%.4f"|format(user.wallet_balance or 0) }}</div>
          {% if user.last_transaction %}<div class="text-xs text-slate-400 mt-1">Last tx: {{ user.last_transaction }}</div>{% endif %}
        </div>
      </div>

      <div class="mt-6 grid md:grid-cols-2 gap-4">
        <div class="p-4 border rounded">
          <h3 class="font-semibold">Send coins (scan recipient)</h3>
          <div class="mt-3">
            <video id="scan_video" autoplay class="rounded shadow"></video>
          </div>
          <div class="mt-3 flex items-center gap-2">
            <input id="amount" type="number" min="0.01" step="0.0001" placeholder="Amount to send" class="border p-2 rounded w-40"/>
            <button id="sendBtn" onclick="scanAndSend()" class="px-3 py-2 bg-sky-600 text-white rounded">Scan & Send</button>
          </div>
          <div id="smsg" class="mt-2 text-sm text-red-500"></div>
        </div>

        <div class="p-4 border rounded">
          <h3 class="font-semibold">Recent transactions</h3>
          <div class="overflow-x-auto mt-3">
            <table class="w-full text-left">
              <thead>
                <tr class="bg-slate-100"><th class="p-2">ID</th><th class="p-2">From</th><th class="p-2">To</th><th class="p-2">Amt</th><th class="p-2">Time</th></tr>
              </thead>
              <tbody>
                {% if txs %}
                  {% for t in txs %}
                    <tr class="odd:bg-white even:bg-slate-50">
                      <td class="p-2 text-sm">{{ t.id }}</td>
                      <td class="p-2 text-sm">{{ t.sender_name or ('User ' ~ t.sender_id) }}</td>
                      <td class="p-2 text-sm">{{ t.recipient_name or ('User ' ~ t.recipient_id) }}</td>
                      <td class="p-2 text-sm">{{ '%.4f'|format(t.amount) }}</td>
                      <td class="p-2 text-sm">{{ t.created_at.strftime("%Y-%m-%d %H:%M:%S") }}</td>
                    </tr>
                  {% endfor %}
                {% else %}
                  <tr><td colspan=5 class="p-2 text-sm text-slate-500">No transactions</td></tr>
                {% endif %}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

<script>
async function ensureScanCamera(){
  const v = document.getElementById('scan_video');
  if (v.srcObject) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video:true});
    v.srcObject = stream; await v.play();
  } catch(e){
    document.getElementById('smsg').textContent = 'Camera error: ' + e.message;
  }
}
ensureScanCamera();

async function scanAndSend(){
  const btn = document.getElementById('sendBtn');
  const msg = document.getElementById('smsg');
  const amt = parseFloat(document.getElementById('amount').value);
  if (!amt || amt <= 0){ msg.textContent = 'Enter valid amount'; return; }
  btn.disabled = true; msg.textContent = 'Capturing...';
  try{
    const v = document.getElementById('scan_video');
    const canvas = document.createElement('canvas'); canvas.width = v.videoWidth || 320; canvas.height = v.videoHeight || 240;
    const ctx = canvas.getContext('2d'); ctx.drawImage(v,0,0,canvas.width,canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg');
    msg.textContent = 'Sending...';
    const res = await fetch('/scan_send', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ image_data: dataUrl, amount: amt })
    });
    const text = await res.text(); let j;
    try { j = JSON.parse(text); } catch(e){ msg.textContent = 'Server error: ' + text; btn.disabled=false; return; }
    if (j.success){
      msg.textContent = `Sent ${j.amount} coins to ${j.recipient.name}. Your new balance: ${j.sender_balance.toFixed(4)}`;
      document.getElementById('balance').textContent = j.sender_balance.toFixed(4);
      setTimeout(()=>location.reload(),900);
    } else {
      msg.textContent = 'Error: ' + (j.error || j.message || 'unknown');
    }
  } catch(err){
    msg.textContent = 'Camera error: ' + err.message;
  } finally { btn.disabled = false; }
}
</script>
    """, user=user, txs=txs)
    return render_template_string(BASE_TEMPLATE, content=content)

@app.route("/scan_send", methods=["POST"])
def scan_send():
    if "user_id" not in session:
        return jsonify({"success": False, "error": "Login required"}), 401
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON"}), 400
    b64 = (data or {}).get("image_data")
    amount = float((data or {}).get("amount") or 0)
    if not b64 or amount <= 0:
        return jsonify({"success": False, "error": "Missing image or invalid amount"}), 400
    emb, err = compute_embedding_from_b64(b64)
    if err:
        return jsonify({"success": False, "error": err}), 400
    recipient, dist = find_user_by_embedding(emb)
    if recipient is None:
        return jsonify({"success": False, "error": "Recipient not found", "distance": dist}), 404

    sender_id = session["user_id"]
    conn = get_db(); cur = conn.cursor()
    try:
        cur.execute("SELECT wallet_balance FROM users WHERE id=%s FOR UPDATE", (sender_id,))
        row = cur.fetchone()
        if not row:
            conn.rollback(); return jsonify({"success": False, "error": "Sender not found"}), 404
        sender_balance = float(row[0] or 0.0)
        if sender_balance < amount:
            conn.rollback(); return jsonify({"success": False, "error": "Insufficient balance"}), 400
        cur.execute("UPDATE users SET wallet_balance = wallet_balance - %s, last_transaction = NOW() WHERE id=%s", (amount, sender_id))
        cur.execute("UPDATE users SET wallet_balance = wallet_balance + %s, last_transaction = NOW() WHERE id=%s", (amount, recipient["id"]))
        cur.execute("INSERT INTO transactions (sender_id, recipient_id, amount) VALUES (%s,%s,%s)", (sender_id, recipient["id"], amount))
        conn.commit()
        cur.execute("SELECT wallet_balance FROM users WHERE id=%s", (sender_id,))
        new_bal = float(cur.fetchone()[0] or 0.0)
    except Exception as e:
        conn.rollback(); return jsonify({"success": False, "error": f"DB error: {e}"}), 500
    finally:
        cur.close(); conn.close()
    return jsonify({"success": True, "recipient": {"id": recipient["id"], "name": recipient["name"]}, "amount": amount, "sender_balance": new_bal})

@app.route("/check_registered", methods=["POST"])
def check_registered():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"registered": False, "error": "Invalid JSON"}), 400
    b64 = (data or {}).get("image_data")
    if not b64:
        return jsonify({"registered": False, "error": "Missing image_data"}), 400
    emb, err = compute_embedding_from_b64(b64)
    if err:
        return jsonify({"registered": False, "error": err}), 400
    user, dist = find_user_by_embedding(emb)
    if user:
        return jsonify({"registered": True, "user": {"id": user["id"], "name": user["name"]}})
    return jsonify({"registered": False, "distance": dist})

@app.route("/logout")
def logout():
    session.clear(); return redirect("/")

@app.route("/test")
def test(): return "Flask functional test OK"

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
