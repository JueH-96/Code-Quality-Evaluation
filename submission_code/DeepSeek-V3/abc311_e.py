# YOUR CODE HERE
import sys

def main():
    H, W, N = map(int, sys.stdin.readline().split())
    holes = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    
    # Initialize a 2D array to mark the positions of holes
    grid = [[0] * (W + 1) for _ in range(H + 1)]
    for a, b in holes:
        grid[a][b] = 1
    
    # Initialize the DP table
    dp = [[0] * (W + 1) for _ in range(H + 1)]
    
    # Fill the DP table
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            if grid[i][j] == 0:
                if i == 1 or j == 1:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    # Calculate the total number of holeless squares
    total = 0
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            total += dp[i][j]
    
    print(total)

if __name__ == "__main__":
    main()