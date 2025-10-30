# YOUR CODE HERE
import sys
from collections import defaultdict

def dfs(graph, start, visited):
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)

def is_good_graph(graph, x, y, p, q):
    # Add the new edge to the graph
    graph[p].add(q)
    graph[q].add(p)
    
    # Check if there is a path between x and y
    visited = set()
    dfs(graph, x, visited)
    
    # Remove the new edge from the graph
    graph[p].remove(q)
    graph[q].remove(p)
    
    return y not in visited

# Read input
input = sys.stdin.read
data = input().split()
N, M = map(int, data[:2])
edges = [tuple(map(int, data[2 * i + 2:2 * i + 4])) for i in range(M)]
K = int(data[2 * M + 2])
forbidden_pairs = [tuple(map(int, data[2 * M + 2 * i + 3:2 * M + 2 * i + 5])) for i in range(K)]
Q = int(data[2 * M + 2 * K + 3])
queries = [tuple(map(int, data[2 * M + 2 * K + 2 * i + 4:2 * M + 2 * K + 2 * i + 6])) for i in range(Q)]

# Build the graph
graph = defaultdict(set)
for u, v in edges:
    graph[u].add(v)
    graph[v].add(u)

# Answer the queries
for p, q in queries:
    for x, y in forbidden_pairs:
        if not is_good_graph(graph, x, y, p, q):
            print("No")
            break
    else:
        print("Yes")