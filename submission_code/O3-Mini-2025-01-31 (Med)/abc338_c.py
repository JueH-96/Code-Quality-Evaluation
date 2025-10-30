def main():
    import sys, math
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    Q = [int(next(it)) for _ in range(N)]
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]

    # We want to maximize S = x+y where x serves dish A and y serves dish B.
    # The ingredient constraint for ingredient i is:
    #   A_i*x + B_i*y <= Q_i.
    # If we fix S = x+y, we can write y = S - x.
    # Then the constraint becomes:
    #   A_i*x + B_i*(S-x) <= Q_i   =>   (A_i - B_i)*x <= Q_i - B_i*S.
    #
    # For each ingredient i, we analyze:
    #  1) If (A_i - B_i) > 0, then we must have:
    #         x <= (Q_i - B_i*S) / (A_i - B_i)
    #     and in addition Q_i - B_i*S must be non-negative.
    #  2) If (A_i - B_i) < 0, dividing by the negative flips the inequality so:
    #         x >= ceil((Q_i - B_i*S) / (A_i - B_i)).
    #  3) If A_i == B_i, then the inequality requires:
    #         B_i*S <= Q_i.
    #
    # Since x must be an integer in [0, S], we can compute an overall allowed interval for x
    # by taking intersection of these constraints over all ingredients.
    #
    # We then binary search over S (the total number of servings) to find the largest S that is feasible.
    
    # A safe upper bound for S is using the total available grams,
    # because each serving uses at least 1 gram overall.
    lo, hi = 0, sum(Q)
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2  # candidate total servings S
        S = mid
        feasible = True
        # Start with x allowed in [0, S]
        L = 0
        R = S
        for i in range(N):
            d = A[i] - B[i]
            # The inequality: d*x <= Q[i] - B[i]*S
            if d > 0:
                # We also require Q[i] - B[i]*S >= 0.
                if Q[i] - B[i]*S < 0:
                    feasible = False
                    break
                # Upper bound for x from the current ingredient.
                r_bound = (Q[i] - B[i]*S) // d
                R = min(R, r_bound)
            elif d < 0:
                # When d < 0, divide by a negative: the inequality becomes
                #   x >= ceil((Q[i] - B[i]*S) / d)
                lower_bound = math.ceil((Q[i] - B[i]*S) / d)
                L = max(L, lower_bound)
            else:
                # d == 0 means A[i] == B[i] and the inequality becomes B[i]*S <= Q[i].
                if B[i] * S > Q[i]:
                    feasible = False
                    break
        # If the interval for x has an integer solution, then S is feasible.
        if feasible and L <= R and L <= S and R >= 0:
            ans = S
            lo = mid + 1
        else:
            hi = mid - 1
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()