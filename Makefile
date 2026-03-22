.PHONY: install train evaluate simulate test lint convert notebook clean

install:
	pip install -r requirements.txt

train:
	python training/train_marl.py

evaluate:
	python evaluation/evaluate.py

simulate:
	python visualization/swarm_renderer.py

test:
	pytest tests/ -v --color=yes

lint:
	black . --line-length 100 && flake8 . --max-line-length=100 --exclude=.git,__pycache__,notebooks

convert:
	python neuromorphic/ann_to_snn.py

notebook:
	jupyter notebook notebooks/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
