from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin
import sqlite3
import os
from flask import send_from_directory
from src.inference import predict_startup_success
from geopy.geocoders import Nominatim

app = Flask(__name__)
app.secret_key = "super_secret_key"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

DB_PATH = "database/users.db"

geolocator = Nominatim(user_agent="startup_predictor")

def get_lat_lon(city, country):
    location = geolocator.geocode(f"{city}, {country}")
    if location:
        return location.latitude, location.longitude
    return None, None

# -----------------------
# User Model
# -----------------------
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, username, password FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return User(*row)
    return None

# -----------------------
# Routes
# -----------------------
@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (?,?)",
                    (username, password))
        conn.commit()
        conn.close()

        flash("Registration successful. Please login.")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, username, password FROM users WHERE username=? AND password=?",
                    (username, password))
        row = cur.fetchone()
        conn.close()

        if row:
            user = User(*row)
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials")

    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    prediction = None
    probability = None

    def checkbox(name):
        return 1 if request.form.get(name) == "on" else 0

    if request.method == "POST":
        # -------- LOCATION (GLOBAL) --------
        city = request.form.get("city")
        country = request.form.get("country")

        lat, lon = get_lat_lon(city, country)

        if lat is None or lon is None:
            flash("Invalid city or country. Please try again.")
            return render_template("predict.html")

        # -------- BUILD MODEL INPUT --------
        user_input = {
            # Industry flags
            "is_software": checkbox("is_software"),
            "is_web": checkbox("is_web"),
            "is_mobile": checkbox("is_mobile"),
            "is_enterprise": checkbox("is_enterprise"),
            "is_ecommerce": checkbox("is_ecommerce"),
            "is_biotech": checkbox("is_biotech"),
            "is_consulting": checkbox("is_consulting"),
            "is_othercategory": checkbox("is_othercategory"),

            # Auto-generated geo coordinates
            "latitude": lat,
            "longitude": lon
        }

        prediction, probability = predict_startup_success(user_input)

    return render_template(
        "predict.html",
        prediction=prediction,
        probability=probability
    )


@app.route("/xai")
@login_required
def xai():
    return render_template("xai.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))
@app.route("/outputs/<path:filename>")
@login_required
def outputs(filename):
    return send_from_directory("outputs", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)

