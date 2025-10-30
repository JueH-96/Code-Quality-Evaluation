from collections import deque

def bfs(grid, visited, i, j, sea_level):
    queue = deque([(i, j)])
    visited[i][j] = True
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    count = 0

    while queue:
        x, y = queue.popleft()
        count += 1
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and not visited[nx][ny] and grid[nx][ny] > sea_level:
                visited[nx][ny] = True
                queue.append((nx, ny))

    return count

def count_above_sea_level(grid, sea_level):
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]
    count = 0

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] > sea_level and not visited[i][j]:
                count += bfs(grid, visited, i, j, sea_level)

    return count

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    index = 0
    H = int(data[index])
    index += 1
    W = int(data[index])
    index += 1
    Y = int(data[index])
    index += 1

    grid = []
    for i in range(H):
        row = []
        for j in range(W):
            row.append(int(data[index]))
            index += 1
        grid.append(row)

    results = []
    for sea_level in range(1, Y + 1):
        results.append(count_above_sea_level(grid, sea_level))

    for result in results:
        print(result)

if __name__ == "__main__":
    main()