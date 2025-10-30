from collections import defaultdict, deque
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    graph = defaultdict(list)
    for _ in range(M):
        A, B, X, Y = map(int, sys.stdin.readline().split())
        graph[A].append((B, X, Y))
        graph[B].append((A, -X, -Y))

    coordinates = {}
    queue = deque([(1, 0, 0)])
    visited = set([1])
    while queue:
        person, x, y = queue.popleft()
        coordinates[person] = (x, y)
        for neighbor, dx, dy in graph[person]:
            if neighbor not in visited:
                queue.append((neighbor, x + dx, y + dy))
                visited.add(neighbor)

    for i in range(1, N + 1):
        if i not in coordinates:
            print("undecidable")
        else:
            print(*coordinates[i])

if __name__ == "__main__":
    solve()