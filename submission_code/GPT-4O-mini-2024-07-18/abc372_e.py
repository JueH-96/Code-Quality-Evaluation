from sys import stdin, stdout
from collections import defaultdict
import heapq

def main():
    input = stdin.read
    data = input().splitlines()
    
    N, Q = map(int, data[0].split())
    graph = defaultdict(set)
    output = []
    
    for i in range(1, Q + 1):
        query = list(map(int, data[i].split()))
        
        if query[0] == 1:
            # Type 1 query: Add an edge
            _, u, v = query
            graph[u].add(v)
            graph[v].add(u)
        
        elif query[0] == 2:
            # Type 2 query: Find k-th largest connected vertex
            _, v, k = query
            connected_vertices = list(graph[v]) + [v]  # Include the vertex itself
            connected_vertices.sort(reverse=True)  # Sort in descending order
            
            if len(connected_vertices) < k:
                output.append("-1")
            else:
                output.append(str(connected_vertices[k - 1]))  # k-th largest (1-indexed)
    
    stdout.write("
".join(output) + "
")

if __name__ == "__main__":
    main()