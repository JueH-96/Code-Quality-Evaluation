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
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        adj[u].append(v)
        adj[v].append(u)

    visited = [False] * (N + 1)
    components = 0

    for i in range(1, N + 1):
        if not visited[i]:
            components += 1
            q = deque()
            q.append(i)
            visited[i] = True
            while q:
                node = q.popleft()
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        q.append(nei)

    print(M + components - N)

if __name__ == "__main__":
    main()