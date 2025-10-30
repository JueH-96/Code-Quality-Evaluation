import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    grid = data[2:]

    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def bfs(start):
        queue = deque([start])
        visited = set()
        visited.add(start)

        while queue:
            x, y = queue.popleft()

            for dx, dy in directions:
                nx, ny = x, y
                while True:
                    nx += dx
                    ny += dy
                    if grid[nx][ny] == '#':
                        nx -= dx
                        ny -= dy
                        break
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        return visited

    ice_squares = bfs((2, 2))
    print(len(ice_squares))

if __name__ == "__main__":
    main()