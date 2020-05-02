.PHONY: install

install:
	pip install .

.PHONY: test

test:
	pytest --pyargs starbucks_campaigns_analytics
