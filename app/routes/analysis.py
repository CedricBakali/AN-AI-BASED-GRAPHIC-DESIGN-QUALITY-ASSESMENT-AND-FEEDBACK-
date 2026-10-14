from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import os

from app.routes.auth import login_required
from app.services.analysis import analyze_image
from app.models.database import save_analysis


analysis_bp = Blueprint("analysis", __name__)


ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp",
    "bmp",
    "gif"
}

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "static",
    "uploads"
)


def allowed_file(filename):
    """Check whether the uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@analysis_bp.route("/analyze", methods=["POST"])
@login_required
def analyze():

    if "image" not in request.files:
        flash("Please select a design to analyse.")
        return redirect(url_for("dashboard.dashboard"))

    file = request.files["image"]

    if file.filename == "":
        flash("Please select a design to analyse.")
        return redirect(url_for("dashboard.dashboard"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload an image file.")
        return redirect(url_for("dashboard.dashboard"))

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = secure_filename(file.filename)

    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(file_path)

    try:
        # Analyse the uploaded design
        result = analyze_image(file_path)

        # Save the assessment for the logged-in user
        analysis_id = save_analysis(
            session["user_id"],
            filename,
            file_path,
            result
        )

        return render_template(
            "results/result.html",
            result=result,
            analysis_id=analysis_id,
            filename=filename
        )

    except Exception as e:
        print("Analysis error:", e)

        flash("Something went wrong while analysing the design.")
        return redirect(url_for("dashboard.dashboard"))