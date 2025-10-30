from itertools import combinations
from collections import deque
import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    M = int(input[idx]); idx += 1
    K = int(input[idx]); idx += 1
    edges = []
    for _ in range(M):
        u = int(input[idx]); idx += 1
        v = int(input[idx]); idx += 1
        w = int(input[idx]); idx += 1
        edges.append((u, v, w))
    
    min_mod = K  # Initialize with a value larger than any possible mod
    
    # Generate all possible combinations of N-1 edges
    for combo in combinations(edges, N-1):
        # Build adjacency list for the current combination
        adj = [[] for _ in range(N + 1)]  # Nodes are 1-based
        for u, v, _ in combo:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check if all nodes are reachable from node 1
        visited = set()
        queue = deque()
        queue.append(1)
        visited.add(1)
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        
        if len(visited) == N:
            # Calculate the total weight sum
            total = sum(w for _, _, w in combo)
            current_mod = total % K
            if current_mod < min_mod:
                min_mod = current_mod
                if min_mod == 0:
                    print(0)
                    return  # Early exit if the minimal possible is found
    
    print(min_mod)

if __name__ == "__main__":
    main()