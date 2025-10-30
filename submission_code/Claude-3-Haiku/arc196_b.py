# Read the input
T = int(input())
for _ in range(T):
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]

    # Count the number of valid placements
    count = 0
    for i in range(4 ** sum(row.count('A') for row in grid) * 2 ** sum(row.count('B') for row in grid)):
        # Check if the current placement is valid
        valid = True
        for x in range(H):
            for y in range(W):
                if grid[x][y] == 'A':
                    if (y == W - 1 and grid[x][0] != grid[x][y]) or (y < W - 1 and grid[x][y + 1] != grid[x][y]):
                        valid = False
                        break
                elif grid[x][y] == 'B':
                    if (x == H - 1 and grid[0][y] != grid[x][y]) or (x < H - 1 and grid[x + 1][y] != grid[x][y]):
                        valid = False
                        break
            if not valid:
                break

        if valid:
            count += 1

    print(count % 998244353)