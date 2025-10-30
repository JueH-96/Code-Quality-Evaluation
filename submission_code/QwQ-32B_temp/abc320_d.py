import sys
from collections import deque

def main():
    input = sys.stdin
    N, M = map(int, input.readline().split())
    adj = [[] for _ in range(N + 1)]  # 1-based indexing

    for _ in range(M):
        A, B, X, Y = map(int, input.readline().split())
        adj[A].append((B, X, Y))
        adj[B].append((A, -X, -Y))

    coords = [None] * (N + 1)
    coords[1] = (0, 0)
    q = deque()
    q.append(1)

    while q:
        current = q.popleft()
        for neighbor, dx, dy in adj[current]:
            if coords[neighbor] is None:
                coords[neighbor] = (coords[current][0] + dx, coords[current][1] + dy)
                q.append(neighbor)

    for i in range(1, N + 1):
        if coords[i] is not None:
            print(coords[i][0], coords[i][1])
        else:
            print("undecidable")

if __name__ == "__main__":
    main()