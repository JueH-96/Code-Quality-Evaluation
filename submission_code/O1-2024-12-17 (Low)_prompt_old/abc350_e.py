def solve():
    import sys
    sys.setrecursionlimit(10**7)
    from functools import lru_cache

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = int(data[1])
    X = float(data[2])
    Y = float(data[3])

    # f(n) = minimum expected cost to reduce n to 0.
    # If n == 0, cost = 0.
    # Otherwise:
    #   costOp1 = X + f(n//A)
    #   costOp2 = (6/5)*Y + (1/5)*(f(n//2) + f(n//3) + f(n//4) + f(n//5) + f(n//6))
    # f(n) = min(costOp1, costOp2)
    #
    # The derivation for costOp2 accounts for the possibility of rolling 1 and staying in n
    # causing repeated costs.  After algebra, the final form of the expected cost if we
    # choose the die operation from state n is (6/5)*Y + (1/5)*Σ_{b=2..6} f(n//b).

    @lru_cache(None)
    def f(n):
        if n == 0:
            return 0.0
        # Operation 1 cost:
        op1 = X + f(n // A)
        # Operation 2 cost:
        # (6/5)*Y + (1/5)*(f(n//2) + f(n//3) + f(n//4) + f(n//5) + f(n//6))
        # do a small optimization by computing partial sums:
        s = f(n // 2) + f(n // 3) + f(n // 4) + f(n // 5) + f(n // 6)
        op2 = (6.0/5.0)*Y + (1.0/5.0)*s
        return min(op1, op2)

    ans = f(N)
    print(f"{ans:.9f}")