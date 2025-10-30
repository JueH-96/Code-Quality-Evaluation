def solve():
    import sys
    import math

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    P = list(map(float, input_data[1:]))

    # Precompute the denominators G(k) = sum_{m=0..k-1} (0.9^m).
    # We also precompute 0.9^m iteratively.
    pow09 = [1.0]*(N+1)
    for i in range(1, N+1):
        pow09[i] = pow09[i-1]*0.9

    G = [0.0]*(N+1)
    for k in range(1, N+1):
        G[k] = G[k-1] + pow09[k-1]

    best_rating = -1e15

    # We will maintain M_previous[i] = max_{1 <= j <= i} dp[j], where dp[j] is
    # the best WeightedSum for the subset of size (k-1) ending at j.
    # Then dp_current[i] = 0.9 * M_previous[i-1] + P[i-1], if i >= k (feasibility).
    # M_current[i] = max(M_current[i-1], dp_current[i])
    # WeightedSum_k = M_current[N].
    # rating_k = WeightedSum_k / G[k] - 1200 / sqrt(k).

    # For k=1, handle separately (no need for M_previous).
    # Then for k=2..N, use the DP relation.

    # M_previous will store maximum dp values for subsets of size (k-1).
    M_previous = [-1e15]*(N+1)  # not really used for k=1, but define for consistency

    # k=1:
    dp_current = [0.0]*(N+1)
    M_current = [-1e15]*(N+1)
    M_current[0] = -1e15
    for i in range(1, N+1):
        dp_current[i] = P[i-1]  # choosing just this contest
        M_current[i] = max(M_current[i-1], dp_current[i])
    # Compute rating
    WeightedSum_1 = M_current[N]
    rating_1 = WeightedSum_1 / G[1] - 1200.0 / math.sqrt(1)
    best_rating = max(best_rating, rating_1)

    # Now M_previous becomes M_current
    M_previous = M_current

    # For k=2..N:
    for k in range(2, N+1):
        dp_current = [0.0]*(N+1)
        M_current = [-1e15]*(N+1)
        M_current[0] = -1e15

        for i in range(1, N+1):
            # We can only form a subset of size k using up to i contests if i >= k
            if i >= k:
                dp_current[i] = 0.9 * M_previous[i-1] + P[i-1]
            else:
                dp_current[i] = -1e15
            M_current[i] = max(M_current[i-1], dp_current[i])

        WeightedSum_k = M_current[N]
        rating_k = WeightedSum_k / G[k] - 1200.0 / math.sqrt(k)
        best_rating = max(best_rating, rating_k)

        M_previous = M_current

    print(f"{best_rating:.9f}")