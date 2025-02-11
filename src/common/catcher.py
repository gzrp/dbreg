import traceback
from datetime import datetime


def exception_catcher(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            sign = '=' * 80 + '\n'
            print(f'{sign}>>>exception time: \t{datetime.now()}\n>>>exception func: \t{func.__name__}\n>>>exception msg: \t{e}')
            print(f'{sign}{traceback.format_exc()}{sign}')
    return wrapper
