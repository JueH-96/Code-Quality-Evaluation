def main():
    import sys
    import math
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = list(map(float, data[1:1+N]))  # Store performances as floats
    
    # Precompute the sum of geometric weights S[k] = 1 + 0.9 + 0.9^2 + ... + 0.9^(k-1)
    S = [0.0] * (N+1)
    c = 1.0
    for k in range(1, N+1):
        S[k] = S[k-1] + c
        c *= 0.9

    # dp[i][k] will be the maximum weighted sum of performances
    # for choosing exactly k contests among the first i contests.
    # However, to save memory, we only store dp for the "previous" i (dp_prev)
    # and for the "current" i (dp_curr).
    dp_prev = [-float('inf')] * (N+1)
    dp_curr = [-float('inf')] * (N+1)
    dp_prev[0] = 0.0  # Choosing 0 contests has weighted sum 0

    # bestK[k] will track the best (maximum) weighted sum achievable
    # using exactly k contests among all i so far.
    bestK = [-float('inf')] * (N+1)
    bestK[0] = 0.0

    for i in range(1, N+1):
        # First copy over the "skip" option (dp_curr[k] = dp_prev[k])
        for k in range(i+1):
            dp_curr[k] = dp_prev[k]
        # Now consider picking the i-th contest (index i-1 in P)
        for k in range(1, i+1):
            val = 0.9 * dp_prev[k-1] + P[i-1]
            if val > dp_curr[k]:
                dp_curr[k] = val
        # Update bestK
        for k in range(1, i+1):
            if dp_curr[k] > bestK[k]:
                bestK[k] = dp_curr[k]
        # For indices beyond i, it's impossible to pick more contests than i
        for k in range(i+1, N+1):
            dp_curr[k] = -float('inf')
        # Swap dp arrays
        dp_prev, dp_curr = dp_curr, dp_prev

    # Find the maximum rating across all possible k >= 1
    answer = -float('inf')
    for k in range(1, N+1):
        # rating = (weighted_sum / S[k]) - 1200 / sqrt(k)
        weighted_sum = bestK[k]
        rating = weighted_sum / S[k] - 1200.0 / math.sqrt(k)
        if rating > answer:
            answer = rating

    # Print the answer with an acceptable precision
    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()