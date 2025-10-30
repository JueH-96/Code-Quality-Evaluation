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
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        adj[a].append(b)
        adj[b].append(a)
    
    visited = [False] * (N + 1)
    ans = 0

    for u in range(1, N + 1):
        if not visited[u]:
            q = deque()
            q.append(u)
            visited[u] = True
            component = [u]
            while q:
                v = q.popleft()
                for neighbor in adj[v]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        component.append(neighbor)
                        q.append(neighbor)
            c = len(component)
            m = 0
            for node in component:
                for neighbor in adj[node]:
                    if visited[neighbor]:
                        m += 1
            m //= 2
            ans += (c * (c - 1) // 2) - m
    print(ans)

if __name__ == '__main__':
    main()