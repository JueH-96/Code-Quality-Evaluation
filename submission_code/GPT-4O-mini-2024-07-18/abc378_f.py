def count_valid_cycles(N, edges):
    from collections import defaultdict

    # Build the adjacency list for the tree
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # Count the degree of each vertex
    degree = [0] * (N + 1)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    # We need to find pairs of vertices that can form a cycle
    # Both vertices in the cycle must have degree 2 in the original tree
    # and must be connected to a third vertex that has degree 2 as well.
    count = 0

    # Check for all pairs of vertices
    for u in range(1, N + 1):
        if degree[u] < 2:
            continue  # Skip vertices with degree less than 2
        for v in graph[u]:
            if degree[v] < 2 or v <= u:
                continue  # Skip if v has degree less than 2 or already processed
            # Check if u and v can form a cycle with a third vertex
            for w in graph[v]:
                if w != u and degree[w] == 2:
                    count += 1

    # Each cycle is counted twice (once for each direction), so divide by 2
    return count // 2

import sys
input = sys.stdin.read

def main():
    data = input().strip().split('
')
    N = int(data[0])
    edges = [tuple(map(int, line.split())) for line in data[1:]]
    
    result = count_valid_cycles(N, edges)
    print(result)

if __name__ == "__main__":
    main()