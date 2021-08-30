import time
from functools import wraps


def timer(func):

    @wraps(func)
    def time_run(*args, **kwargs):

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        total_time = end - start
        print(f"{func.__name__} took {total_time} second(s).")

        return result

    return time_run
