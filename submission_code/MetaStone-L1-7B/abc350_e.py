def main():
    import sys
    N, A, X, Y = map(int, sys.stdin.readline().split())
    memo = {}

    def compute(n):
        if n == 0:
            return 0.0
        if n in memo:
            return memo[n]
        # Compute option1: X + compute(n // A)
        option1 = X + compute(n // A)
        # Compute option2: (6Y + sum(E(n//b) for b in 2-6)) /5
        sum_die = 0
        for b in [2, 3, 4, 5, 6]:
            m = n // b
            sum_die += compute(m)
        option2 = (6 * Y + sum_die) / 5
        # Choose the minimum
        res = min(option1, option2)
        memo[n] = res
        return res

    result = compute(N)
    print("{0:.15f}".format(result))

if __name__ == "__main__":
    main()