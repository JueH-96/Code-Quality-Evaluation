def find_stars(N, edges):
    from collections import defaultdict
    
    # Create a graph from the edges
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Count the degrees of each vertex
    degree_count = {i: 0 for i in range(1, N + 1)}
    for u, v in edges:
        degree_count[u] += 1
        degree_count[v] += 1
    
    # Find the stars
    stars = []
    visited = set()
    
    for node in range(1, N + 1):
        if node not in visited and degree_count[node] > 1:
            # This node is a center of a star
            star_size = degree_count[node]  # The number of edges connected to this node
            stars.append(star_size)
            # Mark all connected nodes as visited
            for neighbor in graph[node]:
                visited.add(neighbor)
            visited.add(node)
    
    # Sort the levels of stars
    stars.sort()
    return stars

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N = int(data[0])
    edges = [tuple(map(int, line.split())) for line in data[1:]]
    
    stars = find_stars(N, edges)
    print(" ".join(map(str, stars)))

if __name__ == "__main__":
    main()