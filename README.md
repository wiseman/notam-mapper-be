# NOTAM Mapper backend

This is a python Flask server that provides an API for the NOTAM Mapper
frontend. See https://github.com/wiseman/notam-mapper-fe for the frontend code.


To run the backend, first set the following environment variables:

| Environment Variable | Description                 |
|----------------------|-----------------------------|
| `OPENAI_API_KEY`     | Your OpenAI API key         |
| `OPENAI_MODEL`       | The name of the model to use|

Then run `python server.py` to start the API server.

