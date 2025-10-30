from functools import lru_cache

@lru_cache(maxsize=None)
def f(n):
    if n <= 1:
        return 0
    if n % 2 == 0:
        return n + 2 * f(n // 2)
    else:
        return n + f(n // 2) + f(n // 2 + 1)

n = int(input().strip())
print(f(n))