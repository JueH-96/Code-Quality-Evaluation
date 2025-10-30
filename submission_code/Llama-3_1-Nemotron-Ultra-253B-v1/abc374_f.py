def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1
    X = int(input[idx]); idx +=1
    T = list(map(int, input[idx:idx+N]))
    
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + T[i]
    
    dp = [{} for _ in range(N+1)]
    dp[0][None] = 0
    
    for i in range(1, N+1):
        current_dp = {}
        start_j = max(0, i - K)
        for j in range(start_j, i):
            for s_prev, cost_prev in dp[j].items():
                group_max_T = T[i-1]
                sum_T = prefix_sum[i] - prefix_sum[j]
                group_size = i - j
                if s_prev is None:
                    new_s = group_max_T
                else:
                    new_s = max(s_prev + X, group_max_T)
                cost = cost_prev + new_s * group_size - sum_T
                if new_s in current_dp:
                    if cost < current_dp[new_s]:
                        current_dp[new_s] = cost
                else:
                    current_dp[new_s] = cost
        dp[i] = current_dp
    
    print(min(dp[N].values()))

if __name__ == "__main__":
    main()