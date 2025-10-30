# YOUR CODE HERE
import sys
from collections import deque

# The problem asks if there exists a sequence X of length N consisting of 0 and 1
# such that X_{A_i} != X_{B_i} for each i=1 to M.
# This condition X_u != X_v means that the elements at positions u and v in sequence X
# must have different values (one must be 0, the other 1).
# We can model this as a graph problem. Create a graph with N vertices, labeled 1 to N.
# For each constraint (A_i, B_i), draw an edge between vertex A_i and vertex B_i.
# The condition X_{A_i} != X_{B_i} is equivalent to requiring that vertices A_i and B_i
# must be assigned different values (colors) from the set {0, 1}.
# This is exactly the definition of a 2-coloring of the graph.
# A graph is 2-colorable if and only if it is bipartite.
# Therefore, the problem reduces to checking if the graph defined by the constraints
# is bipartite.

def solve():
    # Read N and M from the first line of input.
    # N is the number of elements in sequence X (vertices in the graph).
    # M is the number of constraints (edges in the graph).
    # Constraints: 1 <= N, M <= 2 * 10^5
    N, M = map(int, sys.stdin.readline().split())

    # Read sequences A and B from the next two lines.
    # A_i and B_i are 1-based indices (positive integers at most N).
    # Using list comprehension for potentially faster reading of multiple integers
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Step 4: Check for the trivial case where A_i = B_i for some i.
    # If A_i == B_i, the constraint X_{A_i} != X_{B_i} becomes X_{A_i} != X_{A_i},
    # which is impossible for any value of X_{A_i}.
    # So, if any such pair exists, no valid sequence X exists.
    for i in range(M):
        if A[i] == B[i]:
            print("No")
            return # Cannot satisfy the condition, so print "No" and exit.

    # Step 5: Build the adjacency list representation of the graph.
    # Vertices are indexed from 0 to N-1 for convenience in Python lists.
    # The input A_i and B_i are 1-based, so we subtract 1.
    # The graph is undirected, so if there is an edge (u, v), we add v to adj[u]
    # and u to adj[v].
    adj = [[] for _ in range(N)]
    for i in range(M):
        u = A[i] - 1 # Convert 1-based index A_i to 0-based index
        v = B[i] - 1 # Convert 1-based index B_i to 0-based index
        adj[u].append(v)
        adj[v].append(u)

    # Step 6: Initialize the color array for bipartite checking.
    # We will use BFS to check bipartiteness. During BFS, we assign colors
    # (0 or 1) to vertices.
    # color[i] = -1 : Vertex i has not been visited yet.
    # color[i] = 0  : Vertex i has been visited and assigned color 0.
    # color[i] = 1  : Vertex i has been visited and assigned color 1.
    color = [-1] * N

    # Step 7: Perform BFS-based bipartite check.
    # We need to check all connected components of the graph.
    # Iterate through each vertex from 0 to N-1. If a vertex is unvisited,
    # it means it belongs to a new connected component. Start BFS from this vertex.
    is_bipartite = True # Assume the graph is bipartite initially
    for start_node in range(N):
        # If the current vertex `start_node` is unvisited, start a BFS traversal
        # from this node to explore its connected component.
        if color[start_node] == -1:
            # Start BFS for the connected component containing start_node
            q = deque([start_node]) # Initialize the queue with the starting node
            color[start_node] = 0 # Assign the first color (0) to the starting node

            # Perform BFS traversal for the current connected component.
            while q:
                # Dequeue a vertex from the front of the queue.
                u = q.popleft()

                # Explore all neighbors of vertex u.
                for v in adj[u]:
                    if color[v] == -1:
                        # If neighbor v is unvisited, assign it the opposite color of u.
                        # In a bipartite graph, adjacent vertices must have different colors.
                        color[v] = 1 - color[u]
                        # Enqueue the newly visited and colored neighbor to continue BFS.
                        q.append(v)
                    elif color[v] == color[u]:
                        # If neighbor v is already visited and has the same color as u,
                        # this indicates a conflict. An edge connects two vertices of the
                        # same color. This happens only if the graph contains an odd-length
                        # cycle, meaning it is not bipartite.
                        is_bipartite = False
                        # Since we found a conflict, there is no need to continue checking
                        # this component or any other components. We can stop BFS and exit.
                        break # Break from the inner loop iterating over neighbors

                # If a conflict was found during the exploration of neighbors of u,
                # break from the BFS while loop for this component.
                if not is_bipartite:
                    break

        # If a conflict was found in the BFS for the current connected component,
        # break from the outer loop iterating over all possible start_nodes.
        # This optimization prevents unnecessary checks once we know the graph is not bipartite.
        if not is_bipartite:
            break

    # Step 9 & 10: Print the final result based on whether the graph is bipartite.
    # If `is_bipartite` is still True after checking all connected components,
    # it means no conflicts were found, and the graph is bipartite. Thus, a
    # valid sequence X exists.
    if is_bipartite:
        print("Yes")
    # Otherwise, if `is_bipartite` is False, a conflict was found, the graph
    # is not bipartite, and no valid sequence X exists.
    else:
        print("No")

# Execute the solve function to run the program.
solve()