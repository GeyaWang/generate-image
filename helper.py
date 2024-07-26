from time import perf_counter


def timer(func, args=None):
    t1 = perf_counter()

    if args is None:
        out = func()
    else:
        out = func(*args)

    if out is None:
        return perf_counter() - t1
    else:
        return perf_counter() - t1, out
