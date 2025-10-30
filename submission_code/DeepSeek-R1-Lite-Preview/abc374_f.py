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
    for i in range(1, N + 1):
        prefix_sum[i] = prefix_sum[i - 1] + T[i - 1]
    
    # Initialize dp
    INF = float('inf')
    dp = [ (INF, 0) ] * (N + 1)
    dp[0] = (0, -X)  # last_day is set to -X for the first shipment
    
    for i in range(1, N + 1):
        dp[i] = (INF, 0)
        # j ranges from max(0, i-K) to i-1
        start_j = max(0, i - K)
        for j in range(start_j, i):
            # S is the shipping day, which is max(T[i-1], dp[j].last_day + X)
            # T[j] to T[i-1] are the orders in this shipment, and they are sorted
            S = max(T[i - 1], dp[j][1] + X)
            # Calculate dissatisfaction
            diss = dp[j][0] + (i - j) * S - (prefix_sum[i] - prefix_sum[j])
            if diss < dp[i][0]:
                dp[i] = (diss, S)
    
    print(dp[N][0])

if __name__ == '__main__':
    main()