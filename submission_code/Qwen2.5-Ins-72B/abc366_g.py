# YOUR CODE HERE
import sys
from collections import defaultdict

def solve(N, M, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Check if the graph is bipartite
    color = {}
    for i in range(1, N + 1):
        if i not in color:
            color[i] = 0
            stack = [i]
            while stack:
                u = stack.pop()
                for v in graph[u]:
                    if v in color:
                        if color[v] == color[u]:
                            return "No"
                    else:
                        color[v] = 1 - color[u]
                        stack.append(v)
    
    # Assign values to vertices
    values = [1 << (color[i] * 60) for i in range(1, N + 1)]
    return "Yes
" + " ".join(map(str, values))

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
edges = [(int(data[i]), int(data[i + 1])) for i in range(2, 2 * M + 2, 2)]

# Solve and print the result
print(solve(N, M, edges))