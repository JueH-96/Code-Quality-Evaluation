import sys
from collections import deque

def bfs(start, end):
    if (start[0] + start[1]) % 2 != (end[0] + end[1]) % 2:
        return 0
    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)
    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) == end:
            return dist
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
    return 0

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    points = [(int(data[i]), int(data[i + 1])) for i in range(1, 2 * N, 2)]

    total_dist = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            total_dist += bfs(points[i], points[j])

    print(total_dist)

if __name__ == "__main__":
    main()