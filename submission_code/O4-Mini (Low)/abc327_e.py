import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    P = list(map(int, input().split()))

    # DP[j] = maximum weighted sum of performances for a chosen subsequence of length j
    # Weighted sum: if subsequence is Q_1,...,Q_j, then sum_{i=1..j} (0.9)^{j-i} * Q_i
    # Recurrence:
    #   To append P[i] as the j-th pick to some subsequence of length j-1:
    #     new_sum = DP[j-1]*0.9 + P[i]
    # We take the max over all ways to build length j.
    #
    # Denominator for length j is fixed: sum_{t=0..j-1} 0.9^t = (1 - 0.9^j) / 0.1
    # At the end, rating for length j is DP[j]/den_j - 1200/sqrt(j).

    NEG_INF = -1e300
    dp = [NEG_INF] * (N+1)
    dp[0] = 0.0

    for pi in P:
        # iterate j backwards so we don't overwrite dp[j-1] prematurely
        for j in range(N, 0, -1):
            prev = dp[j-1]
            if prev != NEG_INF:
                cand = prev * 0.9 + pi
                if cand > dp[j]:
                    dp[j] = cand

    # Precompute denominators den[j] = sum_{t=0..j-1} 0.9^t
    den = [0.0] * (N+1)
    pow09 = 1.0
    for j in range(1, N+1):
        den[j] = den[j-1] + pow09
        pow09 *= 0.9

    import math
    best = -1e300
    for j in range(1, N+1):
        if dp[j] == NEG_INF:
            continue
        rating = dp[j] / den[j] - 1200.0 / math.sqrt(j)
        if rating > best:
            best = rating

    # print with sufficient precision
    print("{:.15f}".format(best))


if __name__ == "__main__":
    main()