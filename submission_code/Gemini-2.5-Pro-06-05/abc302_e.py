# YOUR CODE HERE
import sys

def solve():
    """
    Solves the graph problem by processing queries and tracking isolated vertices.
    """
    # Use fast I/O for large inputs
    input = sys.stdin.readline

    # Read the number of vertices (N) and queries (Q)
    try:
        line = input()
        if not line:
            return
        N, Q = map(int, line.split())
    except (IOError, ValueError):
        return

    # Adjacency list representation of the graph.
    # Using sets for O(1) average time for adding/removing neighbors.
    adj = [set() for _ in range(N + 1)]

    # `isolated_count` tracks the number of vertices with degree 0.
    # Initially, all N vertices are isolated.
    isolated_count = N

    # Process each query sequentially
    for _ in range(Q):
        try:
            line = input()
            if not line:
                break
            query = list(map(int, line.split()))
        except (IOError, ValueError):
            break

        query_type = query[0]

        if query_type == 1:
            # Type 1: Add an edge between u and v
            u, v = query[1], query[2]

            # If a vertex was isolated (degree 0), it is no longer isolated.
            # `not adj[u]` is an efficient way to check if the neighbor set is empty.
            if not adj[u]:
                isolated_count -= 1
            if not adj[v]:
                isolated_count -= 1

            # Add the edge to the adjacency sets for both vertices.
            adj[u].add(v)
            adj[v].add(u)

        elif query_type == 2:
            # Type 2: Remove all edges connected to vertex v
            v = query[1]

            # This operation only has an effect if v is currently connected.
            # If adj[v] is empty, v is already isolated and nothing changes.
            if adj[v]:
                # Since v had edges and will now have none, it becomes isolated.
                isolated_count += 1

                # We must iterate over a copy of the neighbors, because we will be
                # modifying the graph structure (specifically, the neighbors' adjacency sets).
                neighbors_of_v = list(adj[v])
                for u in neighbors_of_v:
                    # Remove the edge (u, v) by updating u's neighbor set.
                    adj[u].remove(v)
                    # If u has no other edges after this removal, it also becomes isolated.
                    if not adj[u]:
                        isolated_count += 1

                # Clear all of v's edges, officially making its degree 0.
                adj[v].clear()

        # After processing each query, print the current number of isolated vertices.
        print(isolated_count)

solve()