"""Fake News Detection System - Flask application."""
import os, secrets, functools
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, abort)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import db, ml

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", secrets.token_hex(16))
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
UPLOAD_PATH = os.path.join(ml.BASE, "data", "uploaded.csv")
db.init_db()


# ---------- helpers ----------
def login_required(view):
    @functools.wraps(view)
    def wrapped(*a, **k):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*a, **k)
    return wrapped


def admin_required(view):
    @functools.wraps(view)
    def wrapped(*a, **k):
        if not session.get("is_admin"):
            abort(403)
        return view(*a, **k)
    return wrapped


def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


@app.before_request
def check_csrf():
    if request.method == "POST":
        if request.form.get("csrf") != session.get("csrf"):
            abort(400, "Invalid CSRF token")


app.jinja_env.globals["csrf_token"] = csrf_token


# ---------- auth ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        u = request.form["username"].strip()
        e = request.form["email"].strip()
        p = request.form["password"]
        if len(u) < 3 or "@" not in e or len(p) < 6:
            flash("Username >= 3 chars, valid email, password >= 6 chars required.")
            return render_template("register.html")
        conn = db.get_db()
        try:
            conn.execute("INSERT INTO users(username,email,password_hash) VALUES(?,?,?)",
                         (u, e, generate_password_hash(p)))
            conn.commit()
        except Exception:
            flash("Username or email already exists.")
            return render_template("register.html")
        finally:
            conn.close()
        flash("Registered. Please log in.")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        conn = db.get_db()
        user = conn.execute("SELECT * FROM users WHERE username=?",
                            (request.form["username"].strip(),)).fetchone()
        conn.close()
        if user and check_password_hash(user["password_hash"], request.form["password"]):
            session.clear()
            session.update(user_id=user["id"], username=user["username"],
                           is_admin=bool(user["is_admin"]))
            return redirect(url_for("index"))
        flash("Invalid username or password.")
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------- detection ----------
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    result = None
    if request.method == "POST":
        text = request.form["news_text"].strip()
        if len(text) < 20:
            flash("Please enter at least 20 characters of news text.")
        elif not ml.model_exists():
            flash("Model not trained yet. Ask the admin to train it.")
        else:
            label, conf = ml.predict(text)
            conn = db.get_db()
            cur = conn.execute("INSERT INTO news_checks(user_id,news_text) VALUES(?,?)",
                               (session["user_id"], text))
            check_id = cur.lastrowid
            mv = conn.execute("SELECT id FROM model_versions ORDER BY id DESC LIMIT 1").fetchone()
            conn.execute("INSERT INTO predictions(check_id,label,confidence,model_version_id) "
                         "VALUES(?,?,?,?)", (check_id, label, conf, mv["id"] if mv else None))
            conn.commit()
            conn.close()
            result = {"label": label, "confidence": conf, "check_id": check_id}
    return render_template("index.html", result=result)


@app.route("/history")
@login_required
def history():
    conn = db.get_db()
    rows = conn.execute(
        "SELECT n.id, n.news_text, n.created_at, p.label, p.confidence "
        "FROM news_checks n JOIN predictions p ON p.check_id=n.id "
        "WHERE n.user_id=? ORDER BY n.id DESC", (session["user_id"],)).fetchall()
    conn.close()
    return render_template("history.html", rows=rows)


@app.route("/feedback/<int:check_id>", methods=["POST"])
@login_required
def feedback(check_id):
    conn = db.get_db()
    owner = conn.execute("SELECT user_id FROM news_checks WHERE id=?", (check_id,)).fetchone()
    if not owner or owner["user_id"] != session["user_id"]:
        conn.close()
        abort(403)
    conn.execute("INSERT INTO feedback(check_id,user_id,is_correct,comment) VALUES(?,?,?,?)",
                 (check_id, session["user_id"],
                  1 if request.form.get("is_correct") == "1" else 0,
                  request.form.get("comment", "")[:300]))
    conn.commit()
    conn.close()
    flash("Thanks for your feedback!")
    return redirect(url_for("history"))


# ---------- admin ----------
@app.route("/admin", methods=["GET", "POST"])
@login_required
@admin_required
def admin():
    if request.method == "POST":
        f = request.files.get("dataset")
        path = os.path.join(ml.BASE, "data", "sample.csv")
        if f and f.filename:
            if not secure_filename(f.filename).lower().endswith(".csv"):
                flash("Only .csv files are allowed.")
                return redirect(url_for("admin"))
            f.save(UPLOAD_PATH)
            path = UPLOAD_PATH
        try:
            res = ml.train(path)
            conn = db.get_db()
            conn.execute(
                "INSERT INTO model_versions(model_name,rows_used,accuracy,precision_s,recall_s,f1)"
                " VALUES(?,?,?,?,?,?)",
                (res["model_name"], res["rows"], res["accuracy"],
                 res["precision"], res["recall"], res["f1"]))
            conn.commit()
            conn.close()
            flash(f"Training complete. Best model: {res['model_name']} "
                  f"(accuracy {res['accuracy']:.2%}).")
        except Exception as ex:
            flash(f"Training failed: {ex}")
        return redirect(url_for("admin"))
    conn = db.get_db()
    versions = conn.execute("SELECT * FROM model_versions ORDER BY id DESC").fetchall()
    stats = conn.execute(
        "SELECT (SELECT COUNT(*) FROM users) users, "
        "(SELECT COUNT(*) FROM news_checks) checks, "
        "(SELECT COUNT(*) FROM predictions WHERE label='FAKE') fake, "
        "(SELECT COUNT(*) FROM predictions WHERE label='REAL') real").fetchone()
    conn.close()
    return render_template("admin.html", versions=versions, stats=stats)


if __name__ == "__main__":
    app.run(debug=False)
