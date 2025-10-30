import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    conditions = []
    for _ in range(M):
        l, r, x = map(int, sys.stdin.readline().split())
        conditions.append((l, r, x))

    MOD = 998244353

    # Check for impossible conditions (L=R=X). A condition L=R=X requires P_X != P_X, which is impossible.
    # If any such condition exists, no permutation can satisfy it, so the answer is 0.
    for l, r, x in conditions:
        if l == r and r == x:
            print(0)
            return

    # Precompute minR[p][L_prime]: minimum R among conditions (L, R, p) with L < R and L >= L_prime.
    # This helps check if position p is a "bad peak" for an interval [L_prime, j].
    # A position p is a bad peak for interval [i, j] if there's a condition (L_k, R_k, X_k)
    # such that X_k = p, L_k < R_k, and the interval [L_k, R_k] is contained within [i, j].
    # This is equivalent to checking if minR[p][i] <= j.
    # Index p from 1 to N, L_prime from 1 to N. Table size N+2 to handle 1-based indexing and N+1 boundaries.
    minR = [[N + 1] * (N + 2) for _ in range(N + 2)]

    # Group conditions by X
    conds_at_X = [[] for _ in range(N + 1)]
    for l, r, x in conditions:
        # We only care about conditions where the range has size > 1 (L < R).
        # If L=R, the condition is L=R=X, handled already.
        if l < r:
            conds_at_X[x].append((l, r))

    # Build minR table
    for p in range(1, N + 1):
        # Group conditions with X=p by L
        Rs_at_L = [[] for _ in range(N + 1)]
        for l, r in conds_at_X[p]:
            Rs_at_L[l].append(r)

        current_min_R = N + 1
        # Iterate L_prime downwards from p to 1.
        # minR[p][L_prime] is the minimum of minR[p][L_prime+1] and R values for conditions with L=L_prime and X=p.
        for L_prime in range(p, 0, -1):
            for r in Rs_at_L[L_prime]:
                current_min_R = min(current_min_R, r)
            minR[p][L_prime] = current_min_R

    # DP state: dp[s][l] is the number of ways to arrange l relative values (e.g., 1..l) into positions s..s+l-1,
    # satisfying all given conditions (L_k, R_k, X_k) where the range [L_k, R_k] is fully contained within [s, s+l-1].
    # The actual values used in the interval [s, s+l-1] don't affect whether a position is a peak relative to others in the range,
    # so the number of valid relative orderings is what matters.
    # dp[s][l] where s is the starting position (1-indexed) and l is the length.
    # Table size (N+2)x(N+1) to handle s from 1 to N+1 and l from 0 to N.
    dp = [[0] * (N + 1) for _ in range(N + 2)]

    # Base case: dp[s][0] = 1 for empty intervals [s, s-1].
    # There's one way to arrange an empty set of values in an empty set of positions.
    for s in range(1, N + 2):
        dp[s][0] = 1

    # Iterate over interval length l
    for l in range(1, N + 1):
        # Iterate over start position s for intervals of length l
        for s in range(1, N - l + 2):
            e = s + l - 1 # End position of interval [s, e]

            # To compute dp[s][l], consider where the maximum value among the l values is placed within [s, e].
            # Suppose the maximum is placed at position p, where s <= p <= e.
            # The constraints relevant to this interval [s, e] are those (L_k, R_k, X_k) where s <= L_k <= X_k <= R_k <= e.
            # If the maximum is placed at p, and p is X_k for some such constraint with L_k < R_k, then P_p is the max in P_{L_k}..P_{R_k},
            # violating the condition. So the maximum value cannot be placed at position p if p is a bad peak for [s, e].
            # p is a bad peak for [s, e] iff minR[p][s] <= e.
            for p in range(s, e + 1):
                # If p is not a bad peak for [s, e] (i.e., minR[p][s] > e), we can place the maximum at p.
                if minR[p][s] > e:
                    # Placing the maximum at p splits the problem into two independent subproblems:
                    # 1. Arranging the p-s values smaller than the maximum into positions [s, p-1]. This interval has length p-s.
                    #    The number of ways is dp[s][p-s].
                    # 2. Arranging the e-p values smaller than the maximum into positions [p+1, e]. This interval has length e-p.
                    #    The number of ways is dp[p+1][e-p].
                    # The total number of ways when the maximum is at p is the product of ways for the two subproblems.
                    dp[s][l] = (dp[s][l] + (long long)dp[s][p - s] * dp[p + 1][e - p]) % MOD

    # The final answer is the number of ways to arrange N values into positions 1..N, which is dp[1][N].
    print(dp[1][N])

solve()