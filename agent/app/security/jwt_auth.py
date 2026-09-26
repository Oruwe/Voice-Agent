"""
JWT creation and verification for the token service.

Signs with HS256 using JWT_SECRET. Every token carries:
  sub         — the user's external_id (stable identity, not a DB UUID)
  tenant_slug — so the token service can re-validate the tenant without
                a separate lookup before the Postgres call
  iss         — must match JWT_ISSUER to prevent cross-service reuse
  exp         — enforced on decode; expired tokens are rejected hard
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import jwt

TOKEN_EXPIRY_SECONDS = 3600  # 1 hour


class AuthError(Exception):
    """Raised when JWT creation or verification fails. Never leaks which
    specific check failed to external callers — that detail stays in logs."""


@dataclass
class TokenPayload:
    sub: str
    tenant_slug: str
    iss: str
    exp: int


def _secret() -> str:
    secret = os.environ.get("JWT_SECRET", "")
    if not secret:
        raise RuntimeError("JWT_SECRET env var must be set")
    return secret


def _issuer() -> str:
    return os.environ.get("JWT_ISSUER", "voice-agent-platform")


def create_access_token(
    *, subject: str, tenant_slug: str, expires_in: int = TOKEN_EXPIRY_SECONDS
) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "sub": subject,
            "tenant_slug": tenant_slug,
            "iss": _issuer(),
            "iat": now,
            "exp": now + expires_in,
        },
        _secret(),
        algorithm="HS256",
    )


def decode_access_token(token: str) -> TokenPayload:
    try:
        data = jwt.decode(
            token,
            _secret(),
            algorithms=["HS256"],
            issuer=_issuer(),
            options={"require": ["sub", "exp", "iss"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise AuthError("token expired") from e
    except jwt.InvalidIssuerError as e:
        raise AuthError("invalid issuer") from e
    except jwt.InvalidTokenError as e:
        raise AuthError(f"invalid token: {e}") from e

    tenant_slug = data.get("tenant_slug", "")
    if not tenant_slug:
        raise AuthError("missing tenant_slug claim")

    return TokenPayload(
        sub=data["sub"],
        tenant_slug=tenant_slug,
        iss=data["iss"],
        exp=data["exp"],
    )
