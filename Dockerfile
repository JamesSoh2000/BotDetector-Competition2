FROM python:3

RUN pip install requests
RUN pip install pydantic
RUN pip install torch
RUN pip install transformers
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake
RUN pip install sentencepiece==0.1.99

#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]
