def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K, X = map(int, data[:3])
    T = list(map(int, data[3:]))

    # Precompute prefix sums of T to quickly get sums of sub-ranges
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + T[i]

    # dp[i] will hold the minimum total dissatisfaction for orders 1..i
    # dp_day[i] will hold the earliest possible day on which order i's group is shipped
    dp = [float("inf")] * (N + 1)
    dp_day = [0] * (N + 1)

    # Base case: no orders shipped
    dp[0] = 0
    # Setting dp_day[0] to -X so that the first shipping day
    # can be max(dp_day[0] + X, T_1) = max(-X + X, T_1) = T_1
    dp_day[0] = -X

    for i in range(1, N + 1):
        # Try making a group ending exactly at order i of all possible sizes up to K
        for g in range(1, K + 1):
            j = i - g
            if j < 0:
                break
            # The shipping day if we group orders (j+1) to i:
            # Must be at least dp_day[j] + X, and also at least T[i-1] (the largest T in that range)
            shipping_day = max(dp_day[j] + X, T[i - 1])

            # The total dissatisfaction from these g orders:
            # g * shipping_day - sum(T_(j+1..i))
            group_dissatisfaction = g * shipping_day - (prefix_sum[i] - prefix_sum[j])
            candidate = dp[j] + group_dissatisfaction

            # Update dp[i] and dp_day[i] if better
            if candidate < dp[i]:
                dp[i] = candidate
                dp_day[i] = shipping_day
            elif candidate == dp[i] and shipping_day < dp_day[i]:
                # If dissatisfaction ties, use the earlier shipping day
                dp_day[i] = shipping_day

    # The answer is dp[N] – the minimum total dissatisfaction for all orders
    print(dp[N])

# Do not forget to call main()!
if __name__ == "__main__":
    main()