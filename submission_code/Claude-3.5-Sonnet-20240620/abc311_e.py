# YOUR CODE HERE
def count_holeless_squares(H, W, holes):
    grid = [[0] * (W + 1) for _ in range(H + 1)]
    for a, b in holes:
        grid[a][b] = 1

    # Compute prefix sum
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            grid[i][j] += grid[i-1][j] + grid[i][j-1] - grid[i-1][j-1]

    count = 0
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            for n in range(1, min(H - i + 2, W - j + 2)):
                if grid[i+n-1][j+n-1] - grid[i-1][j+n-1] - grid[i+n-1][j-1] + grid[i-1][j-1] == 0:
                    count += 1
                else:
                    break
    return count

H, W, N = map(int, input().split())
holes = [tuple(map(int, input().split())) for _ in range(N)]

result = count_holeless_squares(H, W, holes)
print(result)