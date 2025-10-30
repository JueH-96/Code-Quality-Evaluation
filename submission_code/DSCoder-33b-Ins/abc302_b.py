# YOUR CODE HERE
H, W = map(int, input().split())
grid = [list(input()) for _ in range(H)]

def find_sequence(grid, H, W):
    sequence = ['s', 'n', 'u', 'k', 'e']
    for i in range(H):
        for j in range(W):
            if grid[i][j] == sequence[0]:
                for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    if all(0 <= i + k * di < H and 0 <= j + k * dj < W and grid[i + k * di][j + k * dj] == sequence[k] for k in range(5)):
                        return [(i + k * di + 1, j + k * dj + 1) for k in range(5)]

positions = find_sequence(grid, H, W)
for position in positions:
    print(position[0], position[1])