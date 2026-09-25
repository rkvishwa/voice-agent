"""HTTP auth routes and session cookie tests."""

import unittest

from fastapi.testclient import TestClient

from app.auth import SESSION_COOKIE, verify_credentials
from app.server import app


class TestVerifyCredentials(unittest.TestCase):
    def test_valid_credentials(self) -> None:
        self.assertTrue(
            verify_credentials("pytest@example.com", "pytest-password")
        )

    def test_wrong_password(self) -> None:
        self.assertFalse(
            verify_credentials("pytest@example.com", "wrong")
        )

    def test_wrong_email(self) -> None:
        self.assertFalse(
            verify_credentials("other@example.com", "pytest-password")
        )


class TestAuthRoutes(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_login_failure(self) -> None:
        res = self.client.post(
            "/api/login",
            json={"email": "pytest@example.com", "password": "bad"},
        )
        self.assertEqual(res.status_code, 401)
        self.assertFalse(res.json().get("ok"))
        self.assertNotIn(SESSION_COOKIE, res.cookies)

    def test_login_success_and_protected_route(self) -> None:
        login = self.client.post(
            "/api/login",
            json={"email": "pytest@example.com", "password": "pytest-password"},
        )
        self.assertEqual(login.status_code, 200)
        self.assertTrue(login.json().get("ok"))
        self.assertIn(SESSION_COOKIE, login.cookies)

        authed = TestClient(app, cookies=login.cookies)
        tts = authed.get("/api/tts")
        self.assertEqual(tts.status_code, 200)
        self.assertIn("default_backend", tts.json())

    def test_tts_unauthorized_without_session(self) -> None:
        res = self.client.get("/api/tts")
        self.assertEqual(res.status_code, 401)

    def test_index_redirects_to_login(self) -> None:
        res = self.client.get("/", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertEqual(res.headers.get("location"), "/login")


if __name__ == "__main__":
    unittest.main()
