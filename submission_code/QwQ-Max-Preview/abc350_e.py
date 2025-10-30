import sys
from functools import lru_cache

def main():
    sys.setrecursionlimit(1 << 25)
    N, A, X, Y = map(int, sys.stdin.readline().split())
    memo = {}

    def E(n):
        if n == 0:
            return 0.0
        if n in memo:
            return memo[n]
        # Compute cost1
        cost1 = X + E(n // A)
        # Compute cost2
        sum_others = 0.0
        for b in range(2, 7):
            sum_others += E(n // b)
        cost2 = (6 * Y + sum_others) / 5.0
        res = min(cost1, cost2)
        memo[n] = res
        return res

    result = E(N)
    print("{0:.12f}".format(result))

if __name__ == "__main__":
    main()