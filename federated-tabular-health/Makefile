.PHONY: data baseline federated plot test lint

data:
	python -m src.data_gen

baseline:
	python -m src.run_baseline --epochs 5 --lr 1e-3 --batch-size 64

federated:
	python -m src.run_federated --rounds 5

plot:
	python tools/plot_metrics.py --run_dir $$(ls -td runs/* | head -n1)

test:
	pytest -q

lint:
	black . && isort .
