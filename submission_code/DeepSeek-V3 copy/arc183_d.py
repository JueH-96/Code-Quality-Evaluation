import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        A, B = map(int, sys.stdin.readline().split())
        edges[A].append(B)
        edges[B].append(A)
    
    # Initialize degrees
    degrees = [0] * (N+1)
    for i in range(1, N+1):
        degrees[i] = len(edges[i])
    
    # Initialize queue with leaves
    q = deque()
    for i in range(1, N+1):
        if degrees[i] == 1:
            q.append(i)
    
    # To store the pairs
    pairs = []
    
    while q:
        u = q.popleft()
        if degrees[u] == 0:
            continue
        # Find the neighbor
        for v in edges[u]:
            if degrees[v] > 0:
                break
        # Now, u and v are to be paired
        pairs.append((u, v))
        # Remove u and v
        degrees[u] = 0
        degrees[v] = 0
        # Update degrees of neighbors of v
        for neighbor in edges[v]:
            if degrees[neighbor] > 0:
                degrees[neighbor] -= 1
                if degrees[neighbor] == 1:
                    q.append(neighbor)
    
    # Now, calculate the distances
    # Precompute parent and depth for each node
    parent = [0] * (N+1)
    depth = [0] * (N+1)
    visited = [False] * (N+1)
    q = deque()
    q.append(1)
    visited[1] = True
    while q:
        u = q.popleft()
        for v in edges[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = depth[u] + 1
                q.append(v)
    
    # Function to find LCA
    def lca(u, v):
        while depth[u] > depth[v]:
            u = parent[u]
        while depth[v] > depth[u]:
            v = parent[v]
        while u != v:
            u = parent[u]
            v = parent[v]
        return u
    
    # Function to compute distance
    def distance(u, v):
        return depth[u] + depth[v] - 2 * depth[lca(u, v)]
    
    # Now, print the pairs in the order that maximizes the total distance
    # Since the order doesn't affect the total distance, we can print them in any order
    for u, v in pairs:
        print(u, v)

if __name__ == "__main__":
    main()