import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        X = int(input[ptr])
        ptr += 1
        Y = int(input[ptr])
        ptr += 1
        adj[A].append((B, X, Y))
        adj[B].append((A, -X, -Y))

    positions = [None] * (N + 1)
    positions[1] = (0, 0)

    q = deque()
    q.append(1)

    while q:
        u = q.popleft()
        for v, dx, dy in adj[u]:
            if positions[v] is None:
                positions[v] = (positions[u][0] + dx, positions[u][1] + dy)
                q.append(v)

    for i in range(1, N + 1):
        if positions[i] is not None:
            print(positions[i][0], positions[i][1])
        else:
            print("undecidable")

if __name__ == "__main__":
    main()