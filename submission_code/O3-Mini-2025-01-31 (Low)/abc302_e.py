def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    it = iter(input_data)
    n = int(next(it))
    q = int(next(it))
    
    # Maintain a set for each vertex to store its connected neighbors.
    neighbors = [set() for _ in range(n + 1)]
    # The degree of each vertex.
    degree = [0] * (n + 1)
    # Initially, all vertices are isolated.
    isolated = n

    results = []
    
    for _ in range(q):
        query_type = next(it)
        if query_type == "1":
            # Query of first kind: Add edge between u and v.
            u = int(next(it))
            v = int(next(it))
            # If the vertex was isolated, it ceases to be isolated.
            if degree[u] == 0:
                isolated -= 1
            if degree[v] == 0:
                isolated -= 1
            # Add the edge in both directions.
            neighbors[u].add(v)
            neighbors[v].add(u)
            degree[u] += 1
            degree[v] += 1
        else:
            # Query of second kind: Remove all edges connected to vertex v.
            v = int(next(it))
            if degree[v] > 0:
                # Remove v from all its neighbors.
                for u in list(neighbors[v]):
                    neighbors[u].remove(v)
                    degree[u] -= 1
                    if degree[u] == 0:
                        isolated += 1
                # Clear vertex v's own neighbor set.
                neighbors[v].clear()
                # If vertex v was not isolated, it becomes isolated.
                # Here, degree[v] > 0 means it was not isolated.
                degree[v] = 0
                isolated += 1
        results.append(str(isolated))
    
    sys.stdout.write("
".join(results))

if __name__ == "__main__":
    main()