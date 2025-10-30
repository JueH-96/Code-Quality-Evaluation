def min_vertices_to_delete(N, edges):
    from collections import defaultdict

    # Build the adjacency list for the tree
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # Function to calculate the degree of each vertex
    def calculate_degrees():
        degrees = {}
        for node in graph:
            degrees[node] = len(graph[node])
        return degrees

    degrees = calculate_degrees()

    # Count the number of vertices with degree > 1
    count = 0
    for degree in degrees.values():
        if degree > 1:
            count += 1

    # The number of vertices to delete is the count of vertices with degree > 1 minus 1
    # because we need at least one vertex to remain as the center of the snowflake tree
    return max(0, count - 1)

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N = int(data[0])
    edges = [tuple(map(int, line.split())) for line in data[1:]]
    
    result = min_vertices_to_delete(N, edges)
    print(result)

if __name__ == "__main__":
    main()