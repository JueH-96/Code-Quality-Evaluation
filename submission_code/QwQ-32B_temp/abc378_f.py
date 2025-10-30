import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N + 1)]
    degrees = [0] * (N + 1)
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        degrees[u] += 1
        degrees[v] += 1

    # Build new_adj for nodes with degree 3
    new_adj = [[] for _ in range(N + 1)]
    for u in range(1, N + 1):
        if degrees[u] != 3:
            continue
        for v in adj[u]:
            if degrees[v] == 3:
                new_adj[u].append(v)

    # Find connected components in new_adj
    visited = [False] * (N + 1)
    components = []
    for node in range(1, N + 1):
        if degrees[node] == 3 and not visited[node]:
            q = deque()
            q.append(node)
            visited[node] = True
            component = []
            while q:
                current = q.popleft()
                component.append(current)
                for neighbor in new_adj[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        q.append(neighbor)
            components.append(component)

    ans = 0
    for component in components:
        s = set()
        for node in component:
            for neighbor in adj[node]:
                if degrees[neighbor] == 2:
                    s.add(neighbor)
        k = len(s)
        ans += k * (k - 1) // 2

    print(ans)

if __name__ == "__main__":
    main()