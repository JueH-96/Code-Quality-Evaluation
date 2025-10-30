def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1
    X = int(input[idx]); idx +=1
    T = [int(input[i]) for i in range(idx, idx+N)]
    pre_sum = [0]*(N+1)
    for i in range(N):
        pre_sum[i+1] = pre_sum[i] + T[i]
    
    # Initialize dp
    dp = [{} for _ in range(N+1)]
    dp[0][-X] = 0  # initial state
    
    for i in range(1, N+1):
        # j can be from max(0, i-K) to i-1 inclusive
        start_j = max(0, i - K)
        for j in range(start_j, i):
            # Process each prev_ship_day in dp[j]
            if len(dp[j]) == 0:
                continue
            batch_size = i - j
            sum_batch = pre_sum[i] - pre_sum[j]
            for prev_ship_day in dp[j]:
                prev_cost = dp[j][prev_ship_day]
                new_prev_day = prev_ship_day + X
                max_day = T[i-1]
                new_ship_day = max(new_prev_day, max_day)
                current_diss = new_ship_day * batch_size - sum_batch
                total_cost = prev_cost + current_diss
                # Update dp[i]
                if new_ship_day in dp[i]:
                    if total_cost < dp[i][new_ship_day]:
                        dp[i][new_ship_day] = total_cost
                else:
                    dp[i][new_ship_day] = total_cost
        # Prune dp[i]
        items = sorted(dp[i].items(), key=lambda x: x[0])
        pruned = []
        min_cost = float('inf')
        for day, cost in items:
            if cost < min_cost:
                pruned.append( (day, cost) )
                min_cost = cost
        dp[i] = {day: cost for day, cost in pruned}
    
    print(min(dp[N].values()))

if __name__ == '__main__':
    main()