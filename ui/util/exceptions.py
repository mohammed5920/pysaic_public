class UIException(Exception):
    """generic UI related exception"""
    pass

class UIMediaException(UIException):
    """Can't open / stream media"""
    pass

class UILayoutException(UIException):
    """Can't process a layout"""
    pass

class UIRenderingException(UIException):
    """Error happened while rendering pygame surface"""
    pass