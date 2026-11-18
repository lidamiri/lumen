class LumenError(Exception):
    pass


class ConfigurationError(LumenError):
    pass


class InputError(LumenError):
    pass


class NotFoundError(LumenError):
    pass


class ToolError(LumenError):
    pass


class ProviderError(LumenError):
    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
