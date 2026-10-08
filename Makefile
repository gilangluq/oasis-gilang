.PHONY: install demo run test
install:
	pip install -r requirements-dev.txt
demo:
	python scripts/generate_demo_workbook.py
run:
	streamlit run app.py
test:
	python -m pytest -q tests
