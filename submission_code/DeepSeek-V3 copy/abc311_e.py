def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    
    holes = set()
    index = 3
    for _ in range(N):
        a = int(data[index])
        b = int(data[index+1])
        holes.add((a, b))
        index += 2
    
    # Initialize DP table
    dp = [[0] * (W + 1) for _ in range(H + 1)]
    
    # Fill DP table
    for i in range(1, H+1):
        for j in range(1, W+1):
            if (i, j) not in holes:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    # Calculate the total number of holeless squares
    total = 0
    for i in range(1, H+1):
        for j in range(1, W+1):
            total += dp[i][j]
    
    print(total)

if __name__ == "__main__":
    main()