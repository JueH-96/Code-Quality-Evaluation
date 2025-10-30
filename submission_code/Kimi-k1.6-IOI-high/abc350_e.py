import sys
from functools import lru_cache

def main():
    N, A, X, Y = map(int, sys.stdin.readline().split())
    
    @lru_cache(maxsize=None)
    def expected(n):
        if n == 0:
            return 0.0
        # Option 1: Use the deterministic operation
        next_n1 = n // A
        option1 = X + expected(next_n1)
        # Option 2: Use the die roll
        sum_b = 0.0
        for b in range(2, 7):
            next_nb = n // b
            sum_b += expected(next_nb)
        option2 = (6 * Y + sum_b) / 5.0
        return min(option1, option2)
    
    result = expected(N)
    # Print with sufficient precision
    print("{0:.15f}".format(result))
    
if __name__ == "__main__":
    main()