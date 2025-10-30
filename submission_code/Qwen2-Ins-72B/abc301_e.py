from collections import deque

def bfs(grid, start, goal, walls, candies, T):
    H, W = len(grid), len(grid[0])
    visited = [[[False for _ in range(T+1)] for _ in range(W)] for _ in range(H)]
    queue = deque([(start, 0, 0)])
    visited[start[0]][start[1]][0] = True

    while queue:
        (x, y), steps, candy_count = queue.popleft()

        if (x, y) == goal:
            return candy_count

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and not visited[nx][ny][steps + 1]:
                if grid[nx][ny] == 'o':
                    new_candy_count = candy_count + 1
                else:
                    new_candy_count = candy_count

                if new_candy_count <= len(candies):
                    visited[nx][ny][steps + 1] = True
                    queue.append(((nx, ny), steps + 1, new_candy_count))

    return -1

def main():
    H, W, T = map(int, input().split())
    grid = [list(input()) for _ in range(H)]
    start, goal, walls, candies = None, None, set(), set()

    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)
            elif grid[i][j] == '#':
                walls.add((i, j))
            elif grid[i][j] == 'o':
                candies.add((i, j))

    result = bfs(grid, start, goal, walls, candies, T)
    print(result)

if __name__ == "__main__":
    main()