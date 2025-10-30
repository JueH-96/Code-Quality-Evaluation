import sys
from functools import lru_cache

n = int(sys.stdin.readline())

@lru_cache(maxsize=None)
def calculate(x):
    if x == 1:
        return 0
    if x % 2 == 0:
        return x + 2 * calculate(x // 2)
    else:
        return x + calculate((x - 1) // 2) + calculate((x + 1) // 2)

print(calculate(n))