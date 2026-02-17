from datetime import datetime, timedelta

from flask import Blueprint, Flask, flash, redirect, render_template, request, session, url_for

from idea_bot.config import AppConfig
from idea_bot.repositories.idea_repository import IdeaRepository
from idea_bot.workflows.idea_workflow import IdeaWorkflow


def register_routes(
    app: Flask,
    *,
    config: AppConfig,
    repository: IdeaRepository,
    workflow: IdeaWorkflow,
) -> None:
    bp = Blueprint("web", __name__)

    @bp.route("/", methods=["GET", "POST"])
    def submit_idea():
        if request.method == "POST":
            result = workflow.run(
                name=request.form["name"].strip(),
                email=request.form["email"].strip(),
                title=request.form["title"].strip(),
                description=request.form["description"].strip(),
            )
            return render_template("submit.html", result=result)

        return render_template("submit.html", result=None)

    @bp.route("/manager/login", methods=["GET", "POST"])
    def manager_login():
        if request.method == "POST":
            if (
                request.form.get("username", "") == config.manager_username
                and request.form.get("password", "") == config.manager_password
            ):
                session["manager_authenticated"] = True
                return redirect(url_for("web.manager_dashboard"))
            flash("Invalid credentials", "error")
        return render_template("manager_login.html")

    @bp.route("/manager/logout")
    def manager_logout():
        session.pop("manager_authenticated", None)
        return redirect(url_for("web.manager_login"))

    @bp.route("/manager/dashboard")
    def manager_dashboard():
        if not session.get("manager_authenticated"):
            return redirect(url_for("web.manager_login"))

        ideas = repository.list_ideas()
        now = datetime.utcnow()
        start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_week = now - timedelta(days=7)
        ideas_today = 0
        ideas_week = 0
        categories: dict[str, int] = {}

        for idea in ideas:
            created = datetime.fromisoformat(idea.created_at)
            if created >= start_today:
                ideas_today += 1
            if created >= start_week:
                ideas_week += 1
            categories[idea.category] = categories.get(idea.category, 0) + 1

        return render_template(
            "dashboard.html",
            ideas=ideas,
            total_ideas=len(ideas),
            ideas_today=ideas_today,
            ideas_week=ideas_week,
            categories=categories,
        )

    app.register_blueprint(bp)
