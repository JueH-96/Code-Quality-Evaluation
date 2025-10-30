from functools import lru_cache

@lru_cache(maxsize=None)
def calculate_cost(n):
    if n < 2:
        return 0
    return n + calculate_cost(n // 2) + calculate_cost((n + 1) // 2)

n = int(input())
print(calculate_cost(n))