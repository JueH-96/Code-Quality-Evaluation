import sys
from collections import defaultdict

def process_queries(N, Q, queries):
    # Initialize the graph and the degree of each vertex
    graph = defaultdict(set)
    degree = [0] * (N + 1)
    isolated_count = N  # Initially, all vertices are isolated

    for query in queries:
        if query[0] == 1:
            u, v = query[1], query[2]
            # Connect u and v
            if u not in graph[v] and v not in graph[u]:
                graph[u].add(v)
                graph[v].add(u)
                degree[u] += 1
                degree[v] += 1
                if degree[u] == 1:
                    isolated_count -= 1
                if degree[v] == 1:
                    isolated_count -= 1
        elif query[0] == 2:
            v = query[1]
            # Remove all edges connected to v
            for neighbor in graph[v]:
                graph[neighbor].remove(v)
                degree[neighbor] -= 1
                if degree[neighbor] == 0:
                    isolated_count += 1
            graph[v].clear()
            degree[v] = 0
            isolated_count += 1
        print(isolated_count)

# Read input
input = sys.stdin.read
data = input().split()
N, Q = map(int, [data[0], data[1]])
queries = [[int(data[i]), int(data[i+1]), int(data[i+2])] if data[i] == '1' else [int(data[i]), int(data[i+1])] for i in range(2, len(data), 3)]

# Process queries
process_queries(N, Q, queries)