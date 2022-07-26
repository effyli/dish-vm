class Error(Exception):
    """Base class for other exceptions"""
    pass


class WordNotExistingError(Error):
    """Raised when the word is not in the List"""
    pass
