def main():
    import sys
    import math

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = list(map(int, data[1:]))

    # Precompute the denominator for each possible k (sum of geometric series 1 + 0.9 + 0.9^2 + ... + 0.9^(k-1))
    sumR = [0.0] * (N + 1)
    factor = 1.0
    for k in range(1, N + 1):
        sumR[k] = sumR[k - 1] + factor
        factor *= 0.9

    # Precompute penalty = 1200 / sqrt(k)
    penalty = [0.0] * (N + 1)
    for k in range(1, N + 1):
        penalty[k] = 1200.0 / math.sqrt(k)

    # dp[i][k] will be tracked in a rolling manner:
    # dpPrev[k] = maximum weighted sum of any chosen subsequence of length k using first (i-1) contests
    # dpCurr[k] = maximum weighted sum of any chosen subsequence of length k using first i contests
    dpPrev = [0.0] + [-1e18] * N  # For i=0, picking zero contests => WeightedSum=0, else -inf
    dpCurr = [0.0] * (N + 1)

    best_rating = -1e18

    for i in range(1, N + 1):
        # Contest performance for i-th contest (1-based index, P is 0-based)
        perf_i = P[i - 1]

        dpCurr[0] = 0.0  # Length 0 always has WeightedSum = 0

        # Update DP for lengths 1..i
        for k in range(1, i + 1):
            # Skip i-th contest:
            val_skip = dpPrev[k]
            # Take i-th contest (multiply old sum by 0.9 and add P[i-1]):
            val_take = dpPrev[k - 1] * 0.9 + perf_i

            # Best WeightedSum choosing up to i contests of length k
            best_sum = val_skip if val_skip > val_take else val_take
            dpCurr[k] = best_sum

            # Compute rating for this subsequence
            sr = sumR[k]            # sum of geometric weights
            pen = penalty[k]        # 1200 / sqrt(k)
            rating = best_sum / sr - pen
            if rating > best_rating:
                best_rating = rating

        # Swap dp arrays for the next iteration
        dpPrev, dpCurr = dpCurr, dpPrev

        # Re-initialize dpCurr up to i+1 for next use (avoid re-initializing the entire array if not needed)
        # i+1 can be at most N, so do min(i+2, N+1)
        limit = min(i + 2, N + 1)
        for x in range(limit):
            dpCurr[x] = -1e18

    print(f"{best_rating:.9f}")