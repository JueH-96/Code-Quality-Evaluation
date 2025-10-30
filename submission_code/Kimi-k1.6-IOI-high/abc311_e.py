def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    N = int(data[idx])
    idx += 1
    
    holes = set()
    for _ in range(N):
        a = int(data[idx])
        idx += 1
        b = int(data[idx])
        idx += 1
        holes.add((a, b))
    
    # Initialize DP table with (H+2) rows and (W+2) columns
    dp = [[0] * (W + 2) for _ in range(H + 2)]
    total = 0
    
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            if (i, j) not in holes:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                total += dp[i][j]
            else:
                dp[i][j] = 0
    
    print(total)

if __name__ == "__main__":
    main()