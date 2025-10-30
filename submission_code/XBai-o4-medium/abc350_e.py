import sys

def main():
    import sys
    sys.setrecursionlimit(1000000)
    N, A, X, Y = map(int, sys.stdin.read().split())
    memo = {}

    def dp(n):
        if n == 0:
            return 0.0
        if n in memo:
            return memo[n]
        # Compute option 1: pay X and divide by A
        option1 = X + dp(n // A)
        # Compute option 2: pay Y and roll the die
        sum_others = 0.0
        for b in range(2, 7):
            next_n = n // b
            sum_others += dp(next_n)
        option2 = (6 * Y + sum_others) / 5.0
        # Choose the minimum cost
        res = min(option1, option2)
        memo[n] = res
        return res

    result = dp(N)
    print("{0:.15f}".format(result))

if __name__ == "__main__":
    main()