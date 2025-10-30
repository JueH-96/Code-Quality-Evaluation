import sys
from collections import deque

def main():
    n, m = map(int, sys.stdin.readline().split())
    reversed_adj = [[] for _ in range(n + 1)]
    nodes_from_1 = []
    for _ in range(m):
        a, b = map(int, sys.stdin.readline().split())
        reversed_adj[b].append(a)
        if a == 1:
            nodes_from_1.append(b)
    
    # BFS on reversed graph to find shortest paths to node 1 in original graph
    dist_rev = [float('inf')] * (n + 1)
    dist_rev[1] = 0
    q = deque([1])
    while q:
        u = q.popleft()
        for v in reversed_adj[u]:
            if dist_rev[v] > dist_rev[u] + 1:
                dist_rev[v] = dist_rev[u] + 1
                q.append(v)
    
    min_cycle = float('inf')
    for u in nodes_from_1:
        if dist_rev[u] != float('inf'):
            cycle_len = 1 + dist_rev[u]
            if cycle_len < min_cycle:
                min_cycle = cycle_len
    
    print(-1 if min_cycle == float('inf') else min_cycle)

if __name__ == "__main__":
    main()