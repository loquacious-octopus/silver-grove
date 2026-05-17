# Repository Guidelines

## Project Structure & Module Organization

`pipeline_service/` contains the active FastAPI pipeline service. Core orchestration lives in `pipeline_service/pipeline/`, API schemas in `pipeline_service/schemas/`, configuration loading in `pipeline_service/config/`, and feature modules in `pipeline_service/modules/` such as `scene_coder`, `scene_planner`, `critic`, `judge`, `js_checker`, `renderer`, `monitoring`, and `metrics`. Node-based renderer and validation helpers live under `pipeline_service/modules/renderer/render_service/` and `pipeline_service/modules/js_checker/validator/`. `tests/` contains smoke-test tooling, prompt fixtures, and a simple result viewer. `docker/` contains the Dockerfile, Compose file, Python requirements, and Node dependency lockfile. `abc2/` mirrors much of the repository and should be treated as a snapshot unless a task explicitly targets it.

## Build, Test, and Development Commands

- `pip install -r docker/requirements.txt`: install Python service dependencies.
- `npm install --prefix docker`: install shared Node dependencies from `docker/package.json`.
- `CONFIG_FILE=configuration.yaml python pipeline_service/serve.py`: run the FastAPI service locally from the repo root.
- `bash pipeline_service/run.sh`: run the service with preflight checks and optional local vLLM startup.
- `docker compose -f docker/docker-compose.yml up --build`: build and run the containerized service on port `10006`.
- `python tests/test_pipeline.py --limit 1`: run the HTTP smoke test against a running service.

## Coding Style & Naming Conventions

Use Python 3.11+ style with 4-space indentation, type hints for public interfaces, and `snake_case` for modules, functions, and variables. Keep Pydantic models in `schemas/` and runtime settings in `config/settings.py`. JavaScript files are ES modules (`"type": "module"`); use `camelCase` for functions and variables. Prefer existing module boundaries over adding cross-module utilities.

## Testing Guidelines

Tests are currently smoke and integration oriented. Put new service tests under `tests/` and name Python test files `test_*.py`. The main test script expects the API to be reachable at `localhost:10006` and writes artifacts under the prompt directory, for example `tests/prompts/test/`. For renderer or JS checker changes, exercise both validation and render paths through the pipeline smoke test.

## Commit & Pull Request Guidelines

This repository currently has no commit history, so no established commit convention exists. Use short imperative commit subjects, for example `Add renderer timeout setting` or `Fix pipeline retry state`. Pull requests should describe the behavior change, list commands run, link related issues, and include screenshots or generated artifacts when visual rendering changes.

## Security & Configuration Tips

Do not commit real API keys or model credentials. Use environment variables such as `OPENROUTER_API_KEY`; `configuration.yaml` documents expected keys and local service ports. Generated outputs, caches, and test artifacts should remain out of version control unless they are deliberate fixtures.
