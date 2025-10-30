def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K, X = map(int, data[:3])
    T = list(map(int, data[3:]))

    # We will use 1-based indexing for convenience
    T = [0] + T

    # Prefix sum array to quickly compute sum of T[i..j]
    # prefix[i] = T[1] + T[2] + ... + T[i]
    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix[i] = prefix[i - 1] + T[i]

    # dp[i] will hold the minimum possible dissatisfaction
    # after we have shipped orders 1..i
    # day[i] will hold the shipping day used for the last group
    # that extends the optimal solution for dp[i].
    INF = 10**20
    dp = [INF] * (N + 1)
    day = [0] * (N + 1)

    # For convenience, we set day[0] = -X so that
    # the first time we compute max(day[0] + X, T[i]) we get T[i].
    dp[0] = 0
    day[0] = -X

    # Compute dp[i] using a bottom-up approach
    for i in range(1, N + 1):
        dp[i] = INF
        # Try grouping up to K orders ending with order i
        for group_size in range(1, min(K, i) + 1):
            j = i - group_size
            # candidate_next_day is the day we ship the group (j+1..i)
            # It must be at least day[j] + X (due to the cool-down rule)
            # and also at least T[i] (since T[i] is the largest T among j+1..i)
            candidate_next_day = max(day[j] + X, T[i])

            # cost of shipping this group is:
            # dp[j] plus the dissatisfaction for shipping orders j+1..i
            # dissatisfaction = sum( candidate_next_day - T[k] ) for k in j+1..i
            # = group_size * candidate_next_day - (prefix[i] - prefix[j])
            cost = dp[j] + group_size * candidate_next_day - (prefix[i] - prefix[j])

            if cost < dp[i]:
                dp[i] = cost
                day[i] = candidate_next_day

    # dp[N] is our answer
    print(dp[N])

# Don't forget to call main()
if __name__ == "__main__":
    main()