import sys
from collections import defaultdict

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    edges = []
    for i in range(M):
        u = int(data[2 + 2 * i]) - 1
        v = int(data[3 + 2 * i]) - 1
        edges.append((u, v))
    
    # Build the graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Check if the graph is bipartite
    color = [-1] * N
    def is_bipartite(node, c):
        color[node] = c
        for neighbor in graph[node]:
            if color[neighbor] == -1:
                if not is_bipartite(neighbor, 1 - c):
                    return False
            elif color[neighbor] == c:
                return False
        return True
    
    if not is_bipartite(0, 0):
        print("No")
        return
    
    # Assign values to vertices
    values = [0] * N
    for i in range(N):
        if color[i] == 0:
            values[i] = 1
        else:
            values[i] = 2
    
    # Check the condition
    for i in range(N):
        if len(graph[i]) > 0:
            xor_sum = 0
            for neighbor in graph[i]:
                xor_sum ^= values[neighbor]
            if xor_sum != 0:
                print("No")
                return
    
    # Print the solution
    print("Yes")
    print(" ".join(map(str, values)))

# YOUR CODE HERE