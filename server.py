import json
import os
from typing import Optional

import camel_converter
import flask
from flask_cors import CORS
from gunicorn.app.base import BaseApplication

import notamai

app = flask.Flask(__name__)
CORS(app)

OPENAI_MODEL: Optional[str] = os.environ.get("OPENAI_MODEL")
OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")

if not OPENAI_MODEL:
    raise ValueError("OPENAI_MODEL environment variable must be set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set")


@app.route("/api/parse", methods=["POST"])
def parse():
    notam_text = flask.request.data.decode("utf-8")

    def stream():
        last_notam_dict = None
        for notam in notamai.parse_notam_streaming(OPENAI_MODEL, notam_text):
            notam_dict = notam.model_dump()
            if notam_dict != last_notam_dict:
                last_notam_dict = notam_dict
                notam_json = json.dumps(camel_converter.dict_to_camel(notam_dict))
                yield f"data:{notam_json}\n\n"

    return flask.Response(stream(), content_type="text/event-stream")


class StandaloneApplication(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


if __name__ == "__main__":
    options = {
        "bind": "0.0.0.0:8000",
        "workers": 4,
        "worker_class": "gevent",
        "timeout": 120,
    }
    StandaloneApplication(app, options).run()
