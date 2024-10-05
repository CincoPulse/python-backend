install:
	python -m pip install --upgrade pip &&\
		python -m pip install -r requirements.txt

lint:
	pylint --disable=R,C,W1203,W0702 app.py

activate:
	source venv/bin/activate

venv:
	python3 -m venv venv