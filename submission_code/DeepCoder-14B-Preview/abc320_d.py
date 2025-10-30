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

    x = [None] * (N + 1)
    y = [None] * (N + 1)
    x[1] = 0
    y[1] = 0

    q = deque()
    q.append(1)

    while q:
        u = q.popleft()
        for v, dx, dy in adj[u]:
            if x[v] is None and y[v] is None:
                x[v] = x[u] + dx
                y[v] = y[u] + dy
                q.append(v)

    for i in range(1, N + 1):
        if x[i] is not None and y[i] is not None:
            print(x[i], y[i])
        else:
            print("undecidable")

if __name__ == '__main__':
    main()