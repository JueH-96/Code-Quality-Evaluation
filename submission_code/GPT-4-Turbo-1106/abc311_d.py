from collections import deque

def read_input():
    N, M = map(int, input().split())
    grid = [input().strip() for _ in range(N)]
    return N, M, grid

def bfs(grid, start, N, M):
    visited = set()
    queue = deque([start])
    visited.add(start)
    count = 0

    while queue:
        x, y = queue.popleft()
        count += 1

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x, y
            while True:
                nx += dx
                ny += dy
                if grid[nx][ny] == '#':
                    break
                if (nx, ny) not in visited:
                    queue.append((nx, ny))
                    visited.add((nx, ny))

    return count

def main():
    N, M, grid = read_input()
    result = bfs(grid, (2, 2), N, M)
    print(result)

if __name__ == "__main__":
    main()