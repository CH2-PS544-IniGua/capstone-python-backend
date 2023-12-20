# capstone-python-backend

running the program: 
uvicorn app.main:app
or
python -m app.main

build docker
docker build -t my_fastapi_app .
docker run -d --name my_fastapi_container -p 8080:8080 my_fastapi_app