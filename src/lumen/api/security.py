import hmac

from fastapi import Header, HTTPException, Request, status


def authorize(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    expected = request.app.state.container.settings.api_key
    if not expected:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key")
