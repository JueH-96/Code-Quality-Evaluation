# YOUR CODE HERE
def is_valid(x, y, H, W):
    return 0 <= x < H and 0 <= y < W

def count_possible_positions(H, W, N, T, grid):
    directions = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
    possible_positions = set()

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                x, y = i, j
                valid = True
                for move in T[::-1]:
                    dx, dy = directions[move]
                    x, y = x - dx, y - dy
                    if not is_valid(x, y, H, W) or grid[x][y] == '#':
                        valid = False
                        break
                if valid:
                    possible_positions.add((i, j))

    return len(possible_positions)

H, W, N = map(int, input().split())
T = input().strip()
grid = [input().strip() for _ in range(H)]

result = count_possible_positions(H, W, N, T, grid)
print(result)