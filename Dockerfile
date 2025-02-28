FROM python:3

RUN pip install requests
RUN pip install pydantic
Run pip install torch
Run pip install transformers

#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]
