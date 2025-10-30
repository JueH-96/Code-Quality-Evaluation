import sys
import math
sys.setrecursionlimit(1000000)
from functools import lru_cache

def main():
    N, A, X, Y = map(int, sys.stdin.read().split())

    @lru_cache(maxsize=None)
    def dp(n):
        if n == 0:
            return 0.0
        # Option 1: Pay X and replace n with floor(n / A)
        option1 = X + dp(n // A)
        # Option 2: Pay Y and replace n with floor(n / b) for b=1..6
        sum_option2 = 0.0
        for b in range(1,7):
            sum_option2 += dp(n // b)
        option2 = Y + (sum_option2 / 6.0)
        return min(option1, option2)
    
    result = dp(N)
    print("{0:.12f}".format(result))

if __name__ == "__main__":
    main()