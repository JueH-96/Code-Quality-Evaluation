from math import floor
from functools import lru_cache

N, A, X, Y = map(int, input().split())

@lru_cache(maxsize=None)
def dp(n):
    if n < A:
        return 0
    div_a = X + dp(floor(n / A))
    dice = Y + sum(dp(floor(n / b)) for b in range(1, 7)) / 6
    return min(div_a, dice)

print(dp(N))