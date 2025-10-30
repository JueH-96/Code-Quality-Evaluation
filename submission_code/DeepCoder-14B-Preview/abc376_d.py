import sys
from collections import deque

def bfs(adj, start):
    n = len(adj) - 1  # Nodes are 1-based
    distance = [-1] * (n + 1)
    q = deque()
    q.append(start)
    distance[start] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if distance[v] == -1:
                distance[v] = distance[u] + 1
                q.append(v)
    return distance

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    original_adj = [[] for _ in range(N + 1)]
    reversed_adj = [[] for _ in range(N + 1)]
    edges = []
    
    for _ in range(M):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        original_adj[a].append(b)
        reversed_adj[b].append(a)
        edges.append((a, b))
    
    # Compute forward distances from node 1 in the original graph
    d_forward = bfs(original_adj, 1)
    
    # Compute reversed distances (distance from node to 1 in the original graph)
    d_reversed = bfs(reversed_adj, 1)
    
    min_cycle = float('inf')
    
    for (v, w) in edges:
        if d_forward[v] != -1 and d_reversed[w] != -1:
            current_cycle = d_forward[v] + 1 + d_reversed[w]
            if current_cycle < min_cycle:
                min_cycle = current_cycle
    
    if min_cycle != float('inf'):
        print(min_cycle)
    else:
        print(-1)

if __name__ == '__main__':
    main()