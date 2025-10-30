# YOUR CODE HERE
import sys

def solve():
    # Read N (number of vertices) and Q (number of queries) from standard input.
    N, Q = map(int, sys.stdin.readline().split())

    # Initialize data structures. We use 1-based indexing for vertices (1 to N)
    # for easier mapping from the input vertex numbers.
    
    # 'adj' is a dictionary mapping each vertex ID to a set of its neighbors.
    # Using sets allows for efficient addition and removal of neighbors (average O(1) time).
    adj = {i: set() for i in range(1, N + 1)}
    
    # 'degree' is a list storing the current degree of each vertex. 
    # Index 0 is unused; indices 1 to N correspond to vertices 1 to N. Initialize all degrees to 0.
    degree = [0] * (N + 1) 
    
    # 'isolated_count' tracks the number of vertices that are currently not connected to any other vertex (degree 0).
    # Initially, all N vertices are isolated.
    isolated_count = N

    # 'results' list will store the value of isolated_count after each query.
    results = []

    # Process each of the Q queries.
    for _ in range(Q):
        # Read the query details from standard input.
        query = list(map(int, sys.stdin.readline().split()))
        query_type = query[0] # The type of query (1 or 2).

        if query_type == 1:
            # Query type 1: Add an edge between vertices u and v.
            u, v = query[1], query[2]
            
            # Before adding the edge, check if vertex u was isolated (degree 0).
            # If it was, adding an edge will connect it, so it's no longer isolated.
            # Decrement the isolated_count.
            if degree[u] == 0:
                isolated_count -= 1
            
            # Similarly, check if vertex v was isolated before adding the edge.
            # If it was, it will no longer be isolated. Decrement isolated_count.
            # Note: The problem guarantees u != v. If both u and v were isolated,
            # the count decreases by 2 in total.
            if degree[v] == 0:
                isolated_count -= 1
            
            # Update the graph representation: add v to u's adjacency list and u to v's adjacency list.
            adj[u].add(v)
            adj[v].add(u)
            
            # Increment the degrees of both vertices involved in the new edge.
            degree[u] += 1
            degree[v] += 1
            
        elif query_type == 2:
            # Query type 2: Remove all edges connected to vertex v.
            v = query[1]
            
            # We only need to perform operations if vertex v actually has edges (degree > 0).
            # If degree[v] is already 0, v is isolated and there are no edges connected to it.
            if degree[v] > 0:
                
                # Since vertex v had degree > 0, it was not isolated.
                # After removing all its edges, its degree will become 0, making it isolated.
                # Therefore, increment the isolated_count.
                isolated_count += 1

                # Create a list copy of the neighbors of v.
                # We iterate over this copy because we will modify the adjacency lists during the loop
                # (specifically, removing v from neighbors' lists) and finally clear adj[v].
                # Iterating over a data structure while modifying it can lead to issues.
                neighbors_to_process = list(adj[v]) 

                # Iterate through each neighbor w of vertex v.
                for w in neighbors_to_process:
                    # Remove v from the adjacency list of neighbor w.
                    # Since it's an undirected graph representation, if w is in adj[v], v must be in adj[w].
                    adj[w].remove(v) 
                    
                    # Decrement the degree of neighbor w, as the edge (v, w) is removed.
                    degree[w] -= 1
                    
                    # After decrementing degree[w], check if w has become isolated (degree becomes 0).
                    # If yes, increment the isolated_count.
                    if degree[w] == 0:
                        isolated_count += 1

                # After removing v from all its neighbors' adjacency lists,
                # clear v's own adjacency list completely.
                adj[v].clear()
                # Set the degree of v to 0.
                degree[v] = 0
        
        # After processing the query (of either type 1 or 2), 
        # append the current value of isolated_count to the results list.
        results.append(isolated_count)

    # After processing all Q queries, print the recorded results.
    # Each result corresponds to the state after one query, printed on a new line.
    # Using "
".join(map(str, results)) efficiently creates the required multi-line output string.
    print('
'.join(map(str, results)))

# Call the main function to execute the solution logic.
solve()