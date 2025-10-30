from collections import deque

def bfs(grid, H, W, maxMove):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    visited = [[[False] * (1 << 19) for _ in range(W)] for _ in range(H)]
    queue = deque()

    # Find start and count candies
    candyCount = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                queue.append((i, j, 0, 0))
                visited[i][j][0] = True
            elif grid[i][j] == 'o':
                candyCount += 1

    while queue:
        x, y, moves, bitMask = queue.popleft()

        if grid[x][y] == 'G':
            return bitMask.bit_count() if moves <= maxMove else -1

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and not visited[nx][ny][bitMask]:
                visited[nx][ny][bitMask] = True
                queue.append((nx, ny, moves + 1, bitMask))

                # If the next position is a candy square, set the bit to indicate it's visited
                if grid[nx][ny] == 'o':
                    newBitMask = bitMask | (1 << candyCount.bit_length() - 1)
                    queue.append((nx, ny, moves + 1, newBitMask))

    return -1

# Read input
H, W, T = map(int, input().split())
grid = [input() for _ in range(H)]

result = bfs(grid, H, W, T)
print(result)