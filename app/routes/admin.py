"""Admin routes (Module 10).

Separate from the normal user dashboard. Every route requires
@login_required + @admin_required, enforced by the backend so that
hiding links in HTML is never the only protection.
"""
from __future__ import annotations

from flask import Blueprint, abort, flash, redirect, render_template, session, url_for

from app.models import database
from app.models.user import get_user_by_id, set_user_active
from app.routes.auth import admin_required, login_required


admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.route("/")
@login_required
@admin_required
def dashboard():
    stats = database.get_stats()
    users = database.get_all_users_with_counts()
    recent = database.get_recent_assessments(limit=10)
    return render_template(
        "admin/dashboard.html",
        stats=stats,
        users=users,
        recent=recent,
    )


@admin_bp.route("/users")
@login_required
@admin_required
def users():
    users = database.get_all_users_with_counts()
    return render_template("admin/users.html", users=users)


@admin_bp.route("/users/<int:user_id>")
@login_required
@admin_required
def user_detail(user_id: int):
    user = get_user_by_id(user_id)
    if user is None:
        abort(404)
    assessments = database.get_history(user_id, limit=30)
    return render_template(
        "admin/user_detail.html",
        user=dict(user),
        assessments=assessments,
    )


@admin_bp.route("/users/<int:user_id>/toggle", methods=["POST"])
@login_required
@admin_required
def toggle_user(user_id: int):
    if user_id == session.get("user_id"):
        flash("You cannot disable your own admin account.")
        return redirect(url_for("admin.user_detail", user_id=user_id))
    user = get_user_by_id(user_id)
    if user is None:
        abort(404)
    set_user_active(user_id, not user["is_active"])
    flash(
        f"Account {user['email']} "
        + ("enabled." if not user["is_active"] else "disabled.")
    )
    return redirect(url_for("admin.user_detail", user_id=user_id))
