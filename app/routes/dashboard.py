from flask import Blueprint, render_template, session

from app.routes.auth import login_required
from app.models.user import get_user_by_id
from app.models.database import get_history


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