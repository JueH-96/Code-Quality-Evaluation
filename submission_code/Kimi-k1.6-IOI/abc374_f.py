def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    X = int(input[idx])
    idx += 1
    T = list(map(int, input[idx:idx+N]))
    idx += N
    
    # Compute prefix sums
    prefix_sum = [0] * (N + 1)
    for i in range(1, N+1):
        prefix_sum[i] = prefix_sum[i-1] + T[i-1]
    
    # Initialize DP
    DP = [{} for _ in range(N+1)]
    DP[0] = { -10**18 : 0 }  # Initial state: no shipments, dissatisfaction 0
    
    for i in range(1, N+1):
        start_j = max(0, i - K)
        current_dict = {}
        for j in range(start_j, i):
            if not DP[j]:
                continue
            for prev_day in DP[j]:
                prev_diss = DP[j][prev_day]
                current_T = T[i-1]
                S = max(prev_day + X, current_T)
                cnt = i - j
                sum_T = prefix_sum[i] - prefix_sum[j]
                diss = prev_diss + S * cnt - sum_T
                if S in current_dict:
                    if diss < current_dict[S]:
                        current_dict[S] = diss
                else:
                    current_dict[S] = diss
        DP[i] = current_dict
    
    print(min(DP[N].values()))

if __name__ == "__main__":
    main()