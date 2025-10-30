import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    S = int(input[ptr]) - 1  # converting to 0-based
    ptr += 1
    T = int(input[ptr]) - 1
    ptr += 1

    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(input[ptr]) - 1
        ptr += 1
        v = int(input[ptr]) - 1
        ptr += 1
        adj[u].append(v)
        adj[v].append(u)

    # We need to perform BFS where each state is (a, b)
    # The initial state is (S, T), target is (T, S)
    INF = float('inf')
    dist = [[-1 for _ in range(N)] for __ in range(N)]
    q = deque()
    dist[S][T] = 0
    q.append((S, T))

    found = -1
    while q:
        a, b = q.popleft()
        if a == T and b == S:
            found = dist[a][b]
            break
        # Move A to adjacent nodes
        for na in adj[a]:
            if na != b and dist[na][b] == -1:
                dist[na][b] = dist[a][b] + 1
                q.append((na, b))
        # Move B to adjacent nodes
        for nb in adj[b]:
            if a != nb and dist[a][nb] == -1:
                dist[a][nb] = dist[a][b] + 1
                q.append((a, nb))
    print(found)

if __name__ == '__main__':
    main()