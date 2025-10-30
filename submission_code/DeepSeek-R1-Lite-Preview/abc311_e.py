def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    
    grid = [[0] * (W + 1) for _ in range(H + 1)]
    
    index = 3
    for _ in range(N):
        a = int(data[index])
        b = int(data[index + 1])
        grid[a][b] = 1
        index += 2
    
    dp = [[0] * (W + 1) for _ in range(H + 1)]
    total = 0
    
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            if grid[i][j]:
                dp[i][j] = 0
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
            total += dp[i][j]
    
    print(total)

if __name__ == '__main__':
    main()