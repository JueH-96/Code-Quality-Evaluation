def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, A, X, Y = map(int, input_data)
    from functools import lru_cache

    @lru_cache(None)
    def expected_cost(n):
        # Base case
        if n == 0:
            return 0.0

        # Option 1: Pay X and replace n with floor(n / A)
        cost_det = X + expected_cost(n // A)

        # Option 2: Pay Y, then roll a die (1..6).
        # With probability 1/6, remain at n (cost = expected_cost(n)),
        # and with probability 1/6 for each k in {2..6}, go to floor(n/k).
        # Solving E(n) = Y + 1/6 * [E(n) + sum_{k=2..6} E(floor(n/k))] gives:
        # E(n) = (6/5)*Y + (1/5) * sum_{k=2..6} E(floor(n/k))
        s = (expected_cost(n // 2)
             + expected_cost(n // 3)
             + expected_cost(n // 4)
             + expected_cost(n // 5)
             + expected_cost(n // 6))
        cost_rand = (6.0 * Y + s) / 5.0

        return min(cost_det, cost_rand)

    ans = expected_cost(N)
    print(f"{ans:.9f}")

# Do not forget to call main()!
main()