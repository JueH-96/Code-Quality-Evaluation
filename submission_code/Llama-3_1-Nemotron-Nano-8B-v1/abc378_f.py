import sys
from sys import stdin
from collections import defaultdict, deque

sys.setrecursionlimit(1 << 25)

def main():
    N = int(stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
    
    degree = [0] * (N + 1)
    for u in range(1, N+1):
        degree[u] = len(edges[u])
    
    S = [u for u in range(1, N+1) if degree[u] == 2]
    parent = [0] * (N + 1)
    visited = [False] * (N + 1)
    partner = [0] * (N + 1)
    ans = 0

    for u in S:
        if visited[u]:
            continue
        q = deque()
        q.append(u)
        prev = -1
        found = None
        while q and found is None:
            current = q.popleft()
            next_nodes = [v for v in edges[current] if v != prev]
            if len(next_nodes) != 1:
                continue
            next_node = next_nodes[0]
            if degree[next_node] == 2:
                found = next_node
                break
            if degree[next_node] != 3:
                continue
            prev = current
            q.append(next_node)
        if found is not None and not visited[found]:
            partner[u] = found
            partner[found] = u
            visited[u] = True
            visited[found] = True
            ans += 1

    print(ans)

if __name__ == '__main__':
    main()