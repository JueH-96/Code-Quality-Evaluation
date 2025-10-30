import sys
import itertools
from collections import deque

def main():
    N, M, K = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((u, v, w))
    
    min_mod = K  # Initialize with a value larger than any possible mod
    required = N - 1
    
    for combo in itertools.combinations(edges, required):
        # Build adjacency list
        adj = [[] for _ in range(N + 1)]
        for u, v, w in combo:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check connectivity using BFS starting from node 1
        visited = [False] * (N + 1)
        q = deque()
        q.append(1)
        visited[1] = True
        count = 1
        
        while q:
            node = q.popleft()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    q.append(neighbor)
        
        if count == N:
            # Calculate sum mod K
            total = sum(w for _, _, w in combo)
            current_mod = total % K
            if current_mod < min_mod:
                min_mod = current_mod
    
    print(min_mod)

if __name__ == "__main__":
    main()