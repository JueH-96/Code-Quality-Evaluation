import sys
from functools import lru_cache

@lru_cache(maxsize=None)
def compute_cost(x):
    if x < 2:
        return 0
    a = x // 2
    b = (x + 1) // 2
    return x + compute_cost(a) + compute_cost(b)

n = int(sys.stdin.readline())
print(compute_cost(n))