from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from app.routes.auth import login_required
from app.models.user import check_password, get_user_by_id, update_password
from app.models.database import get_history, get_user_stats


dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/dashboard")
@login_required
def dashboard():

    # Get the logged-in user's ID
    user_id = session["user_id"]

    # Get user information
    user = get_user_by_id(user_id)

    # Get user's assessment history
    assessments = get_history(user_id)

    # Calculate dashboard statistics
    total_designs = len(assessments)
    assessments_completed = len(assessments)

    if assessments:
        average_score = round(
            sum(item["overall"] for item in assessments)
            / len(assessments)
        )
    else:
        average_score = None

    return render_template(
        "dashboard/dashboard.html",
        user=user,
        assessments=assessments,
        total_designs=total_designs,
        assessments_completed=assessments_completed,
        average_score=average_score
    )


# =========================================================
# PROFILE
# =========================================================

@dashboard_bp.route("/profile")
@login_required
def profile():
    """
    The signed-in user's own account page: who they are, how
    their designs are scoring, and password management.
    """

    user_id = session["user_id"]

    user = get_user_by_id(user_id)

    if user is None:
        session.clear()
        flash("Please log in first.")
        return redirect(url_for("auth.login"))

    stats = get_user_stats(user_id)
    recent = get_history(user_id, limit=5)

    return render_template(
        "profile/profile.html",
        user=user,
        stats=stats,
        recent=recent
    )


@dashboard_bp.route("/profile/password", methods=["POST"])
@login_required
def change_password():
    """
    Change the signed-in user's password after re-checking the
    current one, so a stolen session alone cannot lock them out.
    """

    user_id = session["user_id"]

    user = get_user_by_id(user_id)

    if user is None:
        session.clear()
        flash("Please log in first.")
        return redirect(url_for("auth.login"))

    current = request.form.get("current_password", "")
    new_password = request.form.get("new_password", "")
    confirm = request.form.get("confirm_password", "")

    # ---------------------------------------------
    # VALIDATION
    # ---------------------------------------------

    if not check_password(current, user["password_hash"]):
        flash("Your current password is not correct.")
        return redirect(url_for("dashboard.profile"))

    if len(new_password) < 8:
        flash("The new password must be at least 8 characters.")
        return redirect(url_for("dashboard.profile"))

    if new_password != confirm:
        flash("The two new passwords do not match.")
        return redirect(url_for("dashboard.profile"))

    if check_password(new_password, user["password_hash"]):
        flash("Choose a password different from the current one.")
        return redirect(url_for("dashboard.profile"))

    # ---------------------------------------------
    # SAVE
    # ---------------------------------------------

    update_password(user_id, new_password)

    flash("Password updated successfully.")

    return redirect(url_for("dashboard.profile"))