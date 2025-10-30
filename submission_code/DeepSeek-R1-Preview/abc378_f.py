import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u = int(data[idx])
        v = int(data[idx+1])
        adj[u].append(v)
        adj[v].append(u)
        idx += 2

    degree = [0] * (N + 1)
    for u in range(1, N+1):
        degree[u] = len(adj[u])

    visited = [False] * (N + 1)
    total = 0

    for u in range(1, N + 1):
        if not visited[u] and (degree[u] == 2 or degree[u] == 3):
            q = deque()
            q.append(u)
            visited[u] = True
            leaves = 0

            while q:
                current = q.popleft()
                if degree[current] == 2:
                    leaves += 1
                for v in adj[current]:
                    if not visited[v] and (degree[v] == 2 or degree[v] == 3):
                        visited[v] = True
                        q.append(v)

            total += leaves * (leaves - 1) // 2

    print(total)

if __name__ == '__main__':
    main()