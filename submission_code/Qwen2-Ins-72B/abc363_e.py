from collections import deque

def bfs(grid, visited, i, j, h):
    queue = deque([(i, j)])
    visited[i][j] = True
    count = 1 if grid[i][j] > h else 0

    while queue:
        x, y = queue.popleft()

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and not visited[nx][ny]:
                if grid[nx][ny] > h:
                    visited[nx][ny] = True
                    count += 1
                    queue.append((nx, ny))
                elif grid[nx][ny] <= h:
                    visited[nx][ny] = True

    return count

def main():
    H, W, Y = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(H)]

    heights = sorted(set([height for row in grid for height in row]))
    heights.append(10**6)

    ans = []
    prev_area = H * W

    for h in heights:
        visited = [[False] * W for _ in range(H)]
        area = 0

        for i in range(H):
            for j in range(W):
                if not visited[i][j] and grid[i][j] > h:
                    area += bfs(grid, visited, i, j, h)

        prev_area = area
        ans.append(area)

    for i in range(Y):
        print(ans[bisect_left(heights, i + 1)])

if __name__ == "__main__":
    main()