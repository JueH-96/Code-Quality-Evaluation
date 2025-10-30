import sys
from functools import lru_cache

def main():
    N, A, X, Y = map(int, sys.stdin.readline().split())

    @lru_cache(maxsize=None)
    def E(n):
        if n == 0:
            return 0.0
        # Option 1: pay X and divide by A
        m1 = n // A
        cost1 = X + E(m1)
        # Option 2: pay Y and roll die
        sum_b = 0.0
        for b in range(2, 7):
            m = n // b
            sum_b += E(m)
        cost2 = (6 * Y + sum_b) / 5.0
        return min(cost1, cost2)

    ans = E(N)
    print("{0:.12f}".format(ans))

if __name__ == "__main__":
    main()