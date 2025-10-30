import sys

# Use fast I/O
input = sys.stdin.readline
sys.stdout.write

def solve():
    N, Q = map(int, input().split())

    # Use 0-based indexing for vertices
    # Adjacency lists using sets for efficient add/remove operations
    adj = [set() for _ in range(N)]
    # Degree of each vertex
    degree = [0] * N
    # Number of vertices with degree 0 (isolated vertices)
    isolated_count = N

    for _ in range(Q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            # Query type 1: add edge u-v
            u, v = query[1] - 1, query[2] - 1 # Adjust to 0-based indexing

            # If adding the edge causes u or v to stop being isolated,
            # decrement the isolated count.
            # A vertex stops being isolated if its degree becomes non-zero.
            # This happens if its degree was 0 before adding the edge.
            if degree[u] == 0:
                isolated_count -= 1
            if degree[v] == 0:
                isolated_count -= 1
            
            # Add the edge in adjacency lists
            adj[u].add(v)
            adj[v].add(u)
            
            # Increment degrees
            degree[u] += 1
            degree[v] += 1

        elif query[0] == 2:
            # Query type 2: remove all edges incident to v
            v = query[1] - 1 # Adjust to 0-based indexing

            # Get neighbors *before* modifying adj[v] or degrees.
            # Create a list copy because we will modify adj[v] while iterating conceptually.
            neighbors_v = list(adj[v])

            # If removing edges causes v to become isolated,
            # increment the isolated count.
            # v becomes isolated if its degree was > 0 and now becomes 0.
            # Its degree *will* become 0 after this query removes all its edges.
            # So, if degree[v] > 0 currently, it transitions to isolated.
            if degree[v] > 0:
                 isolated_count += 1

            # Process neighbors to remove edges (v, neighbor)
            for neighbor in neighbors_v:
                # Remove the edge (v, neighbor) from the neighbor's adjacency list.
                # The edge will be removed from v's list by adj[v].clear() later.
                adj[neighbor].remove(v)
                
                # Decrement the neighbor's degree.
                degree[neighbor] -= 1
                
                # If the neighbor becomes isolated, increment the isolated count.
                # This check happens after the degree decrement for the neighbor.
                if degree[neighbor] == 0:
                    isolated_count += 1
            
            # After processing all neighbors, set v's degree to 0.
            # We already accounted for v becoming isolated based on its initial degree > 0.
            degree[v] = 0
            
            # Clear v's adjacency list.
            adj[v].clear()

        # Print the number of isolated vertices after each query
        sys.stdout.write(str(isolated_count) + "
")

# Run the solve function
solve()