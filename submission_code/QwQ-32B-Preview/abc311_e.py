import sys

def main():
    data = sys.stdin.read().split()
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    
    holed = []
    for i in range(N):
        a = int(data[2 * i + 3])
        b = int(data[2 * i + 4])
        holed.append((a, b))
    
    # Initialize grid with all cells not holed
    grid = [[1] * (W + 1) for _ in range(H + 1)]
    
    # Mark holed cells
    for a, b in holed:
        grid[a][b] = 0
    
    # Initialize DP table
    dp = [[0] * (W + 1) for _ in range(H + 1)]
    
    # Fill DP table from bottom-right to top-left
    for i in range(H, 0, -1):
        for j in range(W, 0, -1):
            if grid[i][j] == 1:
                dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1])
            else:
                dp[i][j] = 0
    
    # Sum up all values in DP table
    sum_up = sum(dp[i][j] for i in range(1, H + 1) for j in range(1, W + 1))
    
    print(sum_up)

if __name__ == "__main__":
    main()