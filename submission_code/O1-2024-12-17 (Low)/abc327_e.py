def main():
    import sys
    import math

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    P = list(map(int, input_data[1:]))

    # Precompute powers of 0.9 up to N (powers[k] = 0.9^k)
    powers = [1.0] * (N+1)
    for i in range(1, N+1):
        powers[i] = powers[i-1] * 0.9

    # Precompute denominators D[k] = sum_{j=0 to k-1} of 0.9^j
    # which is a geometric series = (1 - 0.9^k)/(1 - 0.9) = 10*(1 - 0.9^k)
    D = [0.0] * (N+1)
    for k in range(1, N+1):
        D[k] = 10.0 * (1 - powers[k])  # sum of 0.9^0 .. 0.9^(k-1)

    # dp[k] = maximum sum of weighted P's when choosing exactly k contests
    # from all we've considered so far.
    dp = [float('-inf')] * (N+1)
    dp[0] = 0.0  # choosing 0 contests yields sum = 0

    for i in range(N):
        # We update dp in descending order of k to avoid overwriting
        for k in range(min(i+1, N), 0, -1):
            if dp[k-1] != float('-inf'):
                val = dp[k-1] + powers[k-1] * P[i]
                if val > dp[k]:
                    dp[k] = val

    # Compute the maximum rating across all 1 <= k <= N
    best_rating = float('-inf')
    for k in range(1, N+1):
        if dp[k] != float('-inf'):
            rating = dp[k] / D[k] - 1200.0 / math.sqrt(k)
            if rating > best_rating:
                best_rating = rating

    print(best_rating)

# Do not forget to call main()
if __name__ == "__main__":
    main()