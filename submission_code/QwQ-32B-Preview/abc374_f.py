def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    X = int(data[2])
    T = list(map(int, data[3:3+N]))
    
    # Initialize dp array with a large number
    dp = [float('inf')] * (N + 1)
    dp[0] = 0  # No dissatisfaction for no orders
    
    for i in range(1, N + 1):
        for j in range(max(0, i - K), i):
            # Shipment day S must be at least the latest T_k and dp[j] + X
            S = max(T[j:i]) if j < i else T[i-1]
            S = max(S, dp[j] + X if j >= 0 else X)
            # Calculate dissatisfaction for shipping orders j+1 to i on day S
            diss = sum(S - T[k] for k in range(j, i))
            dp[i] = min(dp[i], dp[j] + diss)
    
    print(dp[N])

if __name__ == '__main__':
    main()