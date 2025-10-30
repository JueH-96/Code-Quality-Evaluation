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

    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + T[i]

    dp = [{} for _ in range(N+1)]
    dp[0][0] = 0

    for i in range(1, N+1):
        start_j = max(0, i - K)
        for j in range(start_j, i):
            if not dp[j]:
                continue
            for earliest_day, current_sum in dp[j].items():
                ship_day = max(earliest_day, T[i-1])
                num_orders = i - j
                sum_diss = ship_day * num_orders - (prefix[i] - prefix[j])
                new_earliest = ship_day + X
                if new_earliest in dp[i]:
                    if current_sum + sum_diss < dp[i][new_earliest]:
                        dp[i][new_earliest] = current_sum + sum_diss
                else:
                    dp[i][new_earliest] = current_sum + sum_diss

    if dp[N]:
        print(min(dp[N].values()))
    else:
        print(0)

if __name__ == "__main__":
    main()