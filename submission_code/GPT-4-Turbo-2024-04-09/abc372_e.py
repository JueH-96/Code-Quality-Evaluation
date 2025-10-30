import sys
input = sys.stdin.read
from collections import defaultdict

def solve():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    # Using a dictionary to store the connected components for each vertex
    connections = defaultdict(set)
    
    results = []
    
    for _ in range(Q):
        query_type = int(data[index])
        index += 1
        
        if query_type == 1:
            u = int(data[index])
            index += 1
            v = int(data[index])
            index += 1
            # Add each vertex to the other's set
            connections[u].add(v)
            connections[v].add(u)
        elif query_type == 2:
            v = int(data[index])
            index += 1
            k = int(data[index])
            index += 1
            # Get all connected vertices including itself
            connected_vertices = list(connections[v])
            connected_vertices.append(v)
            connected_vertices = sorted(connected_vertices, reverse=True)
            
            if k <= len(connected_vertices):
                results.append(str(connected_vertices[k-1]))
            else:
                results.append("-1")
    
    # Print all results for type 2 queries
    print("
".join(results))