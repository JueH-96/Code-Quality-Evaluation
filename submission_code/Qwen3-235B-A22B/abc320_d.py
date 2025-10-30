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

    pos = [None] * (N + 1)
    pos[1] = (0, 0)
    q = deque([1])

    while q:
        u = q.popleft()
        for v, dx, dy in adj[u]:
            if pos[v] is None:
                x_u, y_u = pos[u]
                pos[v] = (x_u + dx, y_u + dy)
                q.append(v)

    for i in range(1, N + 1):
        if pos[i] is None:
            print("undecidable")
        else:
            print(pos[i][0], pos[i][1])

if __name__ == "__main__":
    main()