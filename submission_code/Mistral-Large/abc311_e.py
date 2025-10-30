import sys

def count_holeless_squares(H, W, N, holes):
    # Initialize a grid to mark holed squares
    grid = [[0] * (W + 1) for _ in range(H + 1)]

    # Mark the holed squares
    for a, b in holes:
        grid[a][b] = 1

    # Initialize a 2D DP array to store the maximum size of holeless square ending at each cell
    dp = [[0] * (W + 1) for _ in range(H + 1)]

    # Fill the DP array
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            if grid[i][j] == 0:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    # Count the number of holeless squares
    count = 0
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            count += dp[i][j]

    return count

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    H = int(data[index])
    W = int(data[index + 1])
    N = int(data[index + 2])
    index += 3

    holes = []
    for _ in range(N):
        a = int(data[index])
        b = int(data[index + 1])
        index += 2
        holes.append((a, b))

    result = count_holeless_squares(H, W, N, holes)
    print(result)

if __name__ == "__main__":
    main()