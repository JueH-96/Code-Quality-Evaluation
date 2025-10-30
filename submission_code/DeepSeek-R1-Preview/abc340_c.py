import sys
from functools import lru_cache

@lru_cache(maxsize=None)
def f(x):
    if x < 2:
        return 0
    a = x // 2
    b = (x + 1) // 2
    return x + f(a) + f(b)

n = int(sys.stdin.readline())
print(f(n))