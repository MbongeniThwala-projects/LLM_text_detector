import re

_PLACEHOLDER = re.compile(r"\[[^\]]*\]")
_WS = re.compile(r"\s+")

def strip_artifacts(text: str) -> str:
    """remove formatting that are not writing style
    
    Drops bracketed placeholders like [Senators Last name] and all whitespace, including newlines, to single spaces
    """
    return _WS.sub(" ", _PLACEHOLDER.sub(" ", text)).strip()