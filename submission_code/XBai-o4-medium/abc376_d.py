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
    edges_to_1 = []

    for _ in range(M):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        adj[a].append(b)
        if b == 1:
            edges_to_1.append(a)

    # BFS to compute distances from node 1
    dist = [-1] * (N + 1)
    q = deque()
    dist[1] = 0
    q.append(1)

    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)

    min_len = float('inf')
    for u in edges_to_1:
        if dist[u] != -1:
            candidate = dist[u] + 1
            if candidate < min_len:
                min_len = candidate

    print(-1 if min_len == float('inf') else min_len)

if __name__ == "__main__":
    main()