def main():
    import sys
    from collections import deque

    # Read input
    input = sys.stdin.read().split()
    N = int(input[0])
    edges = input[1:]
    
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    degree = [0] * (N+1)
    for i in range(0, len(edges), 2):
        u = int(edges[i])
        v = int(edges[i+1])
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
    
    # If vertex 1 is a leaf, operations = 1
    if degree[1] == 1:
        print(1)
        return
    
    # Find all leaves except vertex 1
    leaves = deque()
    for v in range(1, N+1):
        if v != 1 and degree[v] == 1:
            leaves.append(v)
    
    # BFS to find the layer where vertex 1 is reached
    layer = 0
    visited = [False] * (N+1)
    while leaves:
        size = len(leaves)
        for _ in range(size):
            v = leaves.popleft()
            visited[v] = True
            for neighbor in adj[v]:
                if not visited[neighbor]:
                    degree[neighbor] -= 1
                    if degree[neighbor] == 1 and neighbor == 1:
                        # Vertex 1 becomes a leaf in this layer
                        layer += 1
                        print(layer + 1)
                        return
                    elif degree[neighbor] == 1:
                        leaves.append(neighbor)
        layer += 1
    
    # If vertex 1 is not reached, which shouldn't happen in a tree
    print(layer + 1)

if __name__ == "__main__":
    main()