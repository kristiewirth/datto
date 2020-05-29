test:
	pytest -v -p no:warnings

build:
	docker build . -t backend

run: build
	docker run -p 5000:5000 backend 

brew:
	brew install pyenv
	brew install pyenv-virtualenv

pyenv: brew
	pyenv install 3.7.5

poetry: 
	curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

tools: pyenv poetry
