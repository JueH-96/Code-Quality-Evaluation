import sys
from collections import deque

def bfs(x1, y1, x2, y2):
    if (x1 + y1) % 2 != (x2 + y2) % 2:
        return 0

    queue = deque([(x1, y1, 0)])
    visited = set((x1, y1))

    while queue:
        x, y, dist = queue.popleft()

        if x == x2 and y == y2:
            return dist

        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, dist + 1))

    return 0

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    points = []
    index = 1

    for _ in range(N):
        x, y = int(data[index]), int(data[index + 1])
        points.append((x, y))
        index += 2

    total_distance = 0

    for i in range(N - 1):
        for j in range(i + 1, N):
            x1, y1 = points[i]
            x2, y2 = points[j]
            total_distance += bfs(x1, y1, x2, y2)

    print(total_distance)

if __name__ == "__main__":
    main()