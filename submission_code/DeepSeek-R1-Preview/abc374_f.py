def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    X = int(data[2])
    T = list(map(int, data[3:3+N]))
    
    # Compute prefix sums
    prefix_sum = [0] * (N + 1)
    for i in range(1, N+1):
        prefix_sum[i] = prefix_sum[i-1] + T[i-1]
    
    # Initialize DP
    dp = [{} for _ in range(N+1)]
    dp[0][0] = 0  # Starting state: 0 orders processed, next shipment can be on day 0
    
    for i in range(N):
        if not dp[i]:
            continue
        current_dict = dp[i]
        max_j = min(i + K, N)
        for s_prev in list(current_dict.keys()):
            current_diss = current_dict[s_prev]
            for j in range(i+1, max_j + 1):
                # Group is from i to j-1 (0-based)
                group_end = j - 1
                group_T = T[group_end]
                shipment_day = max(s_prev, group_T)
                new_s_prev = shipment_day + X
                count = j - i
                sum_T_group = prefix_sum[j] - prefix_sum[i]
                dissatisfaction = shipment_day * count - sum_T_group
                total = current_diss + dissatisfaction
                
                # Update dp[j]
                if new_s_prev not in dp[j] or total < dp[j].get(new_s_prev, float('inf')):
                    dp[j][new_s_prev] = total
    
    # The minimal dissatisfaction is the minimum value in dp[N]
    print(min(dp[N].values()))

if __name__ == "__main__":
    main()