from flask import Flask

from idea_bot.config import AppConfig
from idea_bot.repositories.idea_repository import IdeaRepository
from idea_bot.services.categorization_service import build_categorizer
from idea_bot.services.embedding_service import FaissIndexStore, build_embedding_services
from idea_bot.services.similarity_service import SimilarityService
from idea_bot.web.routes import register_routes
from idea_bot.workflows.idea_workflow import IdeaWorkflow


def create_app() -> Flask:
    config = AppConfig.from_env()

    repository = IdeaRepository(config)
    repository.init_schema()

    embedding_provider, langchain_embeddings = build_embedding_services(config)
    categorizer = build_categorizer(config)
    similarity_service = SimilarityService()
    faiss_store = FaissIndexStore(config, langchain_embeddings)

    workflow = IdeaWorkflow(
        config=config,
        repository=repository,
        embedding_provider=embedding_provider,
        categorizer=categorizer,
        similarity_service=similarity_service,
        faiss_store=faiss_store,
    )

    app = Flask(
        __name__,
        template_folder=str(config.base_dir / "src" / "idea_bot" / "web" / "templates"),
    )
    app.secret_key = config.secret_key

    register_routes(app, config=config, repository=repository, workflow=workflow)
    return app
