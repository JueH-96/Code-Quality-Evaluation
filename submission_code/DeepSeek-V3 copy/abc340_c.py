def calculate_cost(N):
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(x):
        if x < 2:
            return 0
        return x + dp(x // 2) + dp((x + 1) // 2)

    return dp(N)

# Read input
N = int(input())
# Compute and print the result
print(calculate_cost(N))