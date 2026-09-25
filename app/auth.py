"""Signed-cookie session auth for the web UI."""

import secrets

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.config import AUTH_EMAIL, AUTH_PASSWORD, AUTH_SECRET

SESSION_COOKIE = "voice_agent_session"
SESSION_MAX_AGE = 7 * 24 * 60 * 60

_serializer = URLSafeTimedSerializer(AUTH_SECRET, salt="voice-agent-auth")
_normalized_email = AUTH_EMAIL.strip().lower()


def verify_credentials(email: str, password: str) -> bool:
    if not email or not password:
        return False
    email_ok = secrets.compare_digest(email.strip().lower(), _normalized_email)
    pass_ok = secrets.compare_digest(password, AUTH_PASSWORD)
    return email_ok and pass_ok


def create_session_token() -> str:
    return _serializer.dumps({"sub": _normalized_email})


def parse_session_token(token: str | None) -> bool:
    if not token:
        return False
    try:
        data = _serializer.loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return False
    return isinstance(data, dict) and data.get("sub") == _normalized_email
