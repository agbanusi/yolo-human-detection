from waitress import serve
from paste.translogger import TransLogger
import app
import os

if __name__ == "__main__":
    # serve(app.app, host="0.0.0.0", port=5050)
    # serve(app.app, port=5050)
    # serve(TransLogger(app.app, setup_console_handler=False), host="0.0.0.0", port=5050)
    port = int(os.environ.get('PORT', 5000))
    serve(TransLogger(app.app, setup_console_handler=False), port=port)
