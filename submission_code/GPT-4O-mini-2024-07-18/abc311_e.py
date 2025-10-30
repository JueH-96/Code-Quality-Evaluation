def count_holeless_squares(H, W, holed_squares):
    # Create a grid to mark holed squares
    grid = [[0] * (W + 1) for _ in range(H + 1)]
    
    # Mark the holed squares in the grid
    for a, b in holed_squares:
        grid[a][b] = 1
    
    # Create a dp array to count maximum size of holeless squares
    dp = [[0] * (W + 1) for _ in range(H + 1)]
    
    total_holeless_squares = 0
    
    # Iterate through the grid in reverse order
    for i in range(H, 0, -1):
        for j in range(W, 0, -1):
            if grid[i][j] == 0:  # If the current square is not holed
                # The size of the largest holeless square ending at (i, j)
                dp[i][j] = min(dp[i + 1][j], dp[i][j + 1], dp[i + 1][j + 1]) + 1
                total_holeless_squares += dp[i][j]
    
    return total_holeless_squares

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    H, W, N = map(int, data[0].split())
    holed_squares = [tuple(map(int, line.split())) for line in data[1:N + 1]]
    
    result = count_holeless_squares(H, W, holed_squares)
    print(result)

if __name__ == "__main__":
    main()