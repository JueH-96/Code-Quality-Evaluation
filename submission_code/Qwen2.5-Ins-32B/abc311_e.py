import sys

def count_holeless_squares(H, W, N, holes):
    grid = [[0] * (W + 1) for _ in range(H + 1)]
    for a, b in holes:
        grid[a][b] = 1

    dp = [[0] * (W + 1) for _ in range(H + 1)]
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            if grid[i][j] == 0:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    count = 0
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            count += dp[i][j]
    return count

input = sys.stdin.read
data = input().split()

H, W, N = map(int, data[:3])
holes = [tuple(map(int, data[i:i+2])) for i in range(3, 3+N, 2)]

result = count_holeless_squares(H, W, N, holes)
print(result)