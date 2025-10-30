def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1
    N = int(input[ptr])
    ptr += 1

    # Initialize grid with 0s
    grid = [[0] * W for _ in range(H)]
    for _ in range(N):
        a = int(input[ptr]) - 1  # Convert to 0-based index
        ptr += 1
        b = int(input[ptr]) - 1
        ptr += 1
        grid[a][b] = 1  # Mark as holed

    # Initialize DP table
    dp = [[0] * W for _ in range(H)]

    for i in range(H):
        for j in range(W):
            if grid[i][j] == 1:
                dp[i][j] = 0
            else:
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    # Sum all dp values
    total = 0
    for row in dp:
        total += sum(row)
    print(total)

if __name__ == '__main__':
    main()