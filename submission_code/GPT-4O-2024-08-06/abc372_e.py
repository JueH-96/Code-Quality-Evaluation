# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    # Adjacency list to store the graph
    graph = defaultdict(set)
    
    results = []
    
    for _ in range(Q):
        query_type = int(data[index])
        index += 1
        
        if query_type == 1:
            u = int(data[index])
            index += 1
            v = int(data[index])
            index += 1
            # Add edge between u and v
            graph[u].add(v)
            graph[v].add(u)
        
        elif query_type == 2:
            v = int(data[index])
            index += 1
            k = int(data[index])
            index += 1
            # Get the connected vertices to v
            connected_vertices = list(graph[v])
            # Sort them in descending order
            connected_vertices.sort(reverse=True)
            # Check if we have at least k vertices
            if len(connected_vertices) >= k:
                results.append(connected_vertices[k-1])
            else:
                results.append(-1)
    
    # Print all results for type 2 queries
    for result in results:
        print(result)

if __name__ == "__main__":
    main()