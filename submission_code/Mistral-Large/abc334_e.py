import sys
from collections import deque

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()

    H = int(data[0])
    W = int(data[1])

    grid = [list(data[i]) for i in range(2, H + 2)]

    def count_components():
        visited = [[False] * W for _ in range(H)]
        components = 0

        for i in range(H):
            for j in range(W):
                if grid[i][j] == '#' and not visited[i][j]:
                    components += 1
                    bfs(i, j, visited)

        return components

    def bfs(x, y, visited):
        queue = deque([(x, y)])
        visited[x][y] = True

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#' and not visited[nx][ny]:
                    visited[nx][ny] = True
                    queue.append((nx, ny))

    initial_components = count_components()

    red_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_cells.append((i, j))

    total_change = 0
    total_red_cells = len(red_cells)

    for x, y in red_cells:
        grid[x][y] = '#'
        new_components = count_components()
        total_change += new_components - initial_components
        grid[x][y] = '.'

    expected_value = (initial_components * total_red_cells + total_change) / total_red_cells

    P, Q = expected_value.as_integer_ratio()
    R = (P * pow(Q, MOD - 2, MOD)) % MOD

    print(R)

if __name__ == "__main__":
    main()