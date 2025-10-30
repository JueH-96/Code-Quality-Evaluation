import sys
import collections

def solve():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    edges = data[1:]
    
    # Create adjacency list with edge labels
    graph = [[None] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            char = edges[i][j]
            if char != '-':
                graph[i][j] = char
    
    # Distance matrix initialized to infinity
    dist = [[float('inf')] * N for _ in range(N)]
    
    # Initialize distances for direct edges and self-loops
    for i in range(N):
        for j in range(N):
            if graph[i][j] is not None:
                dist[i][j] = 1
        dist[i][i] = 0  # Distance to self is zero
    
    # Use a modified Floyd-Warshall algorithm to find shortest palindromic paths
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i][k] < float('inf') and dist[k][j] < float('inf'):
                    # Try to extend path i -> k -> j and check if it can be a palindrome
                    if graph[i][k] == graph[j][k]:  # Only consider if they can potentially form a palindrome
                        new_dist = dist[i][k] + dist[k][j]
                        if new_dist < dist[i][j]:
                            dist[i][j] = new_dist
    
    # Print the result matrix
    result = []
    for i in range(N):
        row = []
        for j in range(N):
            if dist[i][j] == float('inf'):
                row.append('-1')
            else:
                row.append(str(dist[i][j]))
        result.append(' '.join(row))
    
    print('
'.join(result))