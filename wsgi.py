"""
WSGI Production Entry Point.

Compatible with:
- Gunicorn (Linux/Container): gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
- Waitress (Windows/Cross-platform): waitress-serve --port=5000 wsgi:app
- Direct execution: python wsgi.py
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fraud_detection.app import create_app
import config

app = create_app()

if __name__ == "__main__":
    print(f"Starting Veritas Risk Ops on {config.HOST}:{config.PORT}...")
    try:
        from waitress import serve
        print("Serving with production Waitress WSGI server...")
        serve(app, host=config.HOST, port=config.PORT)
    except ImportError:
        print("Waitress not available, falling back to Flask development server...")
        app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
