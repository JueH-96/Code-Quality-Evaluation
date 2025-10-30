# YOUR CODE HERE
import sys
input = sys.stdin.read
from collections import defaultdict

data = input().split()
N = int(data[0])
Q = int(data[1])

queries = data[2:]

# To keep track of the number of edges each vertex has
edge_count = [0] * (N + 1)
# To keep track of the number of isolated vertices
isolated_count = N

output = []

i = 0
while i < len(queries):
    query_type = int(queries[i])
    
    if query_type == 1:
        u = int(queries[i + 1])
        v = int(queries[i + 2])
        
        if edge_count[u] == 0:
            isolated_count -= 1
        if edge_count[v] == 0:
            isolated_count -= 1
        
        edge_count[u] += 1
        edge_count[v] += 1
        
        i += 3
    elif query_type == 2:
        v = int(queries[i + 1])
        
        if edge_count[v] > 0:
            isolated_count += 1
        
        isolated_count += edge_count[v]
        edge_count[v] = 0
        
        i += 2
    
    output.append(str(isolated_count))

print("
".join(output))