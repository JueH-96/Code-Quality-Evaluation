from collections import deque

def bfs(grid, n, m):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    queue = deque([(1, 1)])
    visited = set([(1, 1)])
    ice_count = 0

    while queue:
        x, y = queue.popleft()

        if grid[x][y] == '.':
            ice_count += 1
            grid[x][y] = '#'  # Mark as visited

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            while 0 < nx < n and 0 < ny < m and grid[nx][ny] == '.':
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                nx += dx
                ny += dy

    return ice_count

def main():
    n, m = map(int, input().split())
    grid = [list(input()) for _ in range(n)]

    print(bfs(grid, n, m))

if __name__ == "__main__":
    main()