.PHONY: install test smoke validate clean

install:
	python -m pip install -e '.[dev]'

test:
	python -m unittest discover -s tests -v

smoke:
	python examples/synthetic_demo.py --output outputs/synthetic_demo

validate:
	eyeassist validate-data --config configs/analysis.example.yaml

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('outputs').glob('*') if p.is_file() and p.name != '.gitkeep']"
