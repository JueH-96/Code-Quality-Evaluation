def main():
    import sys
    from math import comb
    
    # Read input
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    edges = list(zip(map(int, input[2:2*M+2:2]), map(int, input[3:2*M+3:2])))
    
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for A, B in edges:
        adj[A].append(B)
        adj[B].append(A)
    
    # Initialize visited array
    visited = [False] * (N+1)
    total_operations = 0
    
    # Iterative DFS to find connected components
    for node in range(1, N+1):
        if not visited[node]:
            stack = [node]
            visited[node] = True
            K = 0  # Number of nodes in component
            M_comp = 0  # Number of edges in component
            while stack:
                current = stack.pop()
                K += 1
                for neighbor in adj[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
                    M_comp += 1
            M_comp //= 2  # Since each edge is counted twice
            # Calculate operations for this component
            operations = comb(K, 2) - M_comp
            total_operations += operations
    
    print(total_operations)

if __name__ == "__main__":
    main()