.PHONY: install up down ingest run test demo dev-offline

install:      ## local venv deps
	pip install -r requirements-dev.txt

up:           ## start OpenSearch
	docker compose up -d opensearch

down:
	docker compose down

ingest:       ## load sample_data/*.md into OpenSearch
	python -m scripts.ingest_samples

run:          ## API on :8000  ->  http://localhost:8000/docs
	uvicorn app.main:app --reload --port 8000

test:         ## offline: fake LLM + in-memory store, no keys needed
	pytest -q

demo:         ## one request per route against a running API
	./scripts/demo_requests.sh

stack:        ## everything in docker (api + opensearch)
	docker compose up --build
