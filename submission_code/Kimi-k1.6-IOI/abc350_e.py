import sys
from functools import lru_cache

def main():
    N, A, X, Y = map(int, sys.stdin.readline().split())
    
    @lru_cache(maxsize=None)
    def dp(n):
        if n == 0:
            return 0.0
        # Compute cost for the first operation
        next_n1 = n // A
        cost1 = X + dp(next_n1)
        
        # Compute cost for the second operation
        sum_other = 0.0
        for b in range(2, 7):
            next_n = n // b
            sum_other += dp(next_n)
        cost2 = (6 * Y + sum_other) / 5.0
        
        return min(cost1, cost2)
    
    result = dp(N)
    # Ensure the output has at least 15 decimal places
    print("{0:.15f}".format(result))

if __name__ == "__main__":
    main()