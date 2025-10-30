import numpy as np

def main():
    # Read input
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))
    weights = list(map(int, input().split()))
    pieces = list(map(int, input().split()))
    
    # Build the adjacency list for the graph
    graph = [[] for _ in range(n+1)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # For each vertex, compute the optimal set of neighbors to add pieces to
    optimal_neighbors = [[] for _ in range(n+1)]
    for i in range(1, n+1):
        neighbors = sorted(graph[i], key=lambda x: weights[x-1])  # Sort by weight
        selected = []
        total_weight = 0
        
        for neighbor in neighbors:
            if total_weight + weights[neighbor-1] < weights[i-1]:
                selected.append(neighbor)
                total_weight += weights[neighbor-1]
            else:
                break
        
        optimal_neighbors[i] = selected
    
    # Create matrix A for the equations f = 1 + A * f
    A = np.zeros((n, n))
    for i in range(1, n+1):
        for j in optimal_neighbors[i]:
            A[i-1][j-1] = 1
    
    # Solve the equations (I - A) * f = 1
    I = np.identity(n)
    one = np.ones(n)
    
    f = np.linalg.solve(I - A, one)
    
    # Calculate the total number of operations
    total_operations = sum(pieces[i-1] * f[i-1] for i in range(1, n+1))
    
    print(int(round(total_operations)))

if __name__ == "__main__":
    main()