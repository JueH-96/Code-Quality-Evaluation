import sys
from functools import lru_cache

def main():
    N, A, X, Y = map(int, sys.stdin.readline().split())
    max_recursion = 100000  # To prevent stack overflow for very deep recursion

    @lru_cache(maxsize=None)
    def e(n, memo):
        if n == 0:
            return 0.0
        if n in memo:
            return memo[n]
        
        option1 = X + e(n // A, memo)
        sum_ws = 0.0
        has_self = False
        for b in range(1, 7):
            m = n // b
            if m == n:
                has_self = True
            else:
                sum_ws += e(m, memo)
        
        if has_self:
            if 5 * option1 <= 6 * Y + sum_ws:
                res = option1
            else:
                res = (6 * Y + sum_ws) / 5
        else:
            option2 = Y + sum_ws / 6
            res = min(option1, option2)
        
        memo[n] = res
        return res

    # Use a dictionary for memoization to handle very large N
    memo = {}
    result = e(N, memo)
    print("{0:.15f}".format(result))

if __name__ == "__main__":
    main()