import sys
import math
from functools import lru_cache

def main():
    N, A, X, Y = map(int, sys.stdin.readline().split())
    sum_log = sum(math.log(b) for b in [2, 3, 4, 5, 6]) / math.log(A)
    condition = 6 * Y >= X * sum_log

    if condition:
        steps = 0
        n = N
        while n > 0:
            steps += 1
            n = n // A
        print("{0:.12f}".format(steps * X))
        return
    else:
        memo = {}

        def compute(n):
            if n == 0:
                return 0.0
            if n in memo:
                return memo[n]
            # Option 1 cost
            option1 = X + compute(n // A)
            # Option 2 cost
            sum_b = 0.0
            for b in [2, 3, 4, 5, 6]:
                sum_b += compute(n // b)
            option2 = (6 * Y + sum_b) / 5
            memo[n] = min(option1, option2)
            return memo[n]

        result = compute(N)
        print("{0:.12f}".format(result))

if __name__ == "__main__":
    main()