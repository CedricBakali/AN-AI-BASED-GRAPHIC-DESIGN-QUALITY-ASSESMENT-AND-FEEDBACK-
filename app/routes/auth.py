import sqlite3
from functools import wraps

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session
)

from app.models.user import (
    create_user,
    get_user_by_email,
    check_password
)


# =========================================================
# AUTHENTICATION BLUEPRINT
# =========================================================

auth_bp = Blueprint("auth", __name__)


# =========================================================
# LOGIN REQUIRED DECORATOR
# =========================================================

def login_required(view):
    """
    Protect a route so that only logged-in users
    can access it.
    """

    @wraps(view)
    def wrapped(*args, **kwargs):

        if "user_id" not in session:
            flash("Please log in first.")
            return redirect(url_for("auth.login"))

        return view(*args, **kwargs)

    return wrapped


# =========================================================
# REGISTER
# =========================================================

@auth_bp.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        # Get form data
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # ---------------------------------------------
        # VALIDATION
        # ---------------------------------------------

        if not username:
            flash("Please enter a username.")
            return redirect(url_for("auth.register"))

        if not email or "@" not in email:
            flash("Please enter a valid email address.")
            return redirect(url_for("auth.register"))

        if len(password) < 8:
            flash("Password must be at least 8 characters.")
            return redirect(url_for("auth.register"))

        # ---------------------------------------------
        # CHECK EXISTING ACCOUNT
        # ---------------------------------------------

        if get_user_by_email(email):
            flash("An account with that email already exists.")
            return redirect(url_for("auth.register"))

        # ---------------------------------------------
        # CREATE USER
        # ---------------------------------------------

        try:

            create_user(
                username,
                email,
                password
            )

        except sqlite3.IntegrityError:

            flash("An account with that email already exists.")
            return redirect(url_for("auth.register"))

        # ---------------------------------------------
        # SUCCESS
        # ---------------------------------------------

        flash("Account created successfully. You can now log in.")

        return redirect(url_for("auth.login"))

    return render_template("auth/register.html")


# =========================================================
# LOGIN
# =========================================================

@auth_bp.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        # Get form data
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # ---------------------------------------------
        # FIND USER
        # ---------------------------------------------

        user = get_user_by_email(email)

        if not user:
            flash("Invalid email or password.")
            return redirect(url_for("auth.login"))

        # ---------------------------------------------
        # CHECK PASSWORD
        # ---------------------------------------------

        if not check_password(
            password,
            user["password_hash"]
        ):
            flash("Invalid email or password.")
            return redirect(url_for("auth.login"))

        # ---------------------------------------------
        # CHECK ACCOUNT STATUS
        # ---------------------------------------------

        if not user["is_active"]:
            flash("Your account has been disabled.")
            return redirect(url_for("auth.login"))

        # ---------------------------------------------
        # CREATE LOGIN SESSION
        # ---------------------------------------------

        session.clear()

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["role"] = user["role"]

        # ---------------------------------------------
        # REDIRECT BASED ON ROLE
        # ---------------------------------------------

        if user["role"] == "admin":
            flash("Welcome, Administrator.")
            # No admin blueprint registered yet; fall back to dashboard.
            try:
                return redirect(url_for("admin.dashboard"))
            except Exception:
                return redirect(url_for("dashboard.dashboard"))

        flash("Login successful.")

        return redirect(url_for("dashboard.dashboard"))

    return render_template("auth/login.html")


# =========================================================
# LOGOUT
# =========================================================

@auth_bp.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out.")

    return redirect(url_for("auth.login"))