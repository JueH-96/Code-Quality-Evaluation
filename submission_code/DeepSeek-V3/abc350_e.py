import sys
from functools import lru_cache

def main():
    N, A, X, Y = map(int, sys.stdin.readline().split())
    
    @lru_cache(maxsize=None)
    def dp(n):
        if n == 0:
            return 0.0
        # Option 1: Pay X yen to replace n with floor(n / A)
        cost1 = X + dp(n // A)
        # Option 2: Pay Y yen to roll a die
        # Expected cost is Y + (sum of dp(n // b) for b in 1..6) / 6
        total = 0.0
        for b in range(1, 7):
            total += dp(n // b)
        cost2 = Y + total / 6
        return min(cost1, cost2)
    
    result = dp(N)
    print("{0:.15f}".format(result))

if __name__ == "__main__":
    main()