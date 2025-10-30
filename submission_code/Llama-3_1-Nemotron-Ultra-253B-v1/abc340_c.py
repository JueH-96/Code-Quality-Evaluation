from functools import lru_cache

@lru_cache(maxsize=None)
def compute_sum(x):
    if x == 1:
        return 0
    if x % 2 == 0:
        half = x // 2
        return x + 2 * compute_sum(half)
    else:
        a = x // 2
        b = a + 1
        return x + compute_sum(a) + compute_sum(b)

n = int(input())
print(compute_sum(n))