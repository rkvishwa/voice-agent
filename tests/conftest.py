"""Ensure .env and auth defaults load before app.config is imported."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

os.environ.setdefault("AUTH_EMAIL", "pytest@example.com")
os.environ.setdefault("AUTH_PASSWORD", "pytest-password")
os.environ.setdefault("AUTH_SECRET", "pytest-secret-signing-key")
