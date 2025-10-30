import sys
def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    A = int(data[1])
    X = int(data[2])
    Y = int(data[3])

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(n):
        # Base case
        if n == 0:
            return 0.0
        # Option A: pay X, divide by A
        costA = X + dp(n // A)
        # Option B: roll die until b>=2 succeeds
        # Expected cost = (6*Y + sum_{b=2..6} E(n//b)) / 5
        s = dp(n // 2) + dp(n // 3) + dp(n // 4) + dp(n // 5) + dp(n // 6)
        costB = (6.0 * Y + s) / 5.0
        # Take the cheaper
        return costA if costA < costB else costB

    ans = dp(N)
    # Print with enough precision
    print("{:.15f}".format(ans))

if __name__ == "__main__":
    main()