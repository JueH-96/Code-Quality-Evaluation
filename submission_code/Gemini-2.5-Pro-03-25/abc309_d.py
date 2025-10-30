# YOUR CODE HERE
import sys
from collections import deque

# Function to perform BFS and return distances from start_node
def bfs(start_node, N_total, adj):
    """
    Performs Breadth-First Search starting from start_node on the graph represented by adj.
    
    Args:
        start_node (int): The vertex from which to start the BFS.
        N_total (int): The total number of vertices in the graph.
        adj (list of list of int): The adjacency list representation of the graph.
        
    Returns:
        list of int: A list where dist[i] is the shortest distance (number of edges) 
                     from start_node to vertex i. Unreachable vertices will have a distance of -1.
    """
    # Initialize a queue for BFS and add the start node.
    q = deque([start_node])
    
    # Initialize distances array. Using -1 to denote infinity/unvisited state.
    # The array size is N_total + 1 to accommodate 1-based vertex indexing (index 0 is unused).
    dist = [-1] * (N_total + 1) 
    
    # Distance from the start node to itself is 0.
    dist[start_node] = 0
    
    # Process nodes level by level using the queue.
    while q:
        # Dequeue a vertex u to visit.
        u = q.popleft()
        
        # Explore neighbors of the current node u.
        for v in adj[u]:
            # If neighbor v has not been visited yet (its distance is still -1)
            if dist[v] == -1:
                # Update distance of v: it's one more than the distance to u.
                dist[v] = dist[u] + 1
                # Enqueue v for later processing.
                q.append(v)
                
    # Return the computed distances from start_node to all other vertices.
    return dist

def solve():
    """
    Reads graph description, computes the required maximum shortest path distance, and prints the result.
    """
    # Read N1, N2 (sizes of the two vertex sets) and M (number of edges) from standard input.
    N1, N2, M = map(int, sys.stdin.readline().split())
    
    # Calculate the total number of vertices in the graph.
    N_total = N1 + N2
    
    # Initialize adjacency list representation of the graph.
    # Using a list of lists, size N_total + 1 for 1-based vertex indexing.
    adj = [[] for _ in range(N_total + 1)]
    
    # Read M edges from standard input and build the adjacency list.
    # The graph is undirected, so for an edge (u, v), add v to u's list and u to v's list.
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # Perform BFS starting from vertex 1. 
    # This finds shortest distances from vertex 1 to all reachable vertices.
    # Due to problem guarantees, this includes all vertices in V1 = {1, ..., N1}.
    dist1 = bfs(1, N_total, adj)
    
    # Find the maximum distance from vertex 1 to any vertex in V1.
    # This value is denoted as D1. Initialize it to 0.
    D1 = 0
    # Iterate through vertices from 1 to N1.
    for i in range(1, N1 + 1):
        # Problem guarantees ensure V1 is connected and contains vertex 1,
        # so all vertices i in {1, ..., N1} are reachable from 1. dist1[i] will be non-negative.
        # Update D1 with the maximum distance found so far.
        # Add a safety check for reachability, although theoretically unnecessary given guarantees.
        if dist1[i] != -1: 
           D1 = max(D1, dist1[i])
        # else: pass # This case should not happen based on problem statement guarantees.

    # Perform BFS starting from vertex N_total (the last vertex, N = N1 + N2).
    # This finds shortest distances from vertex N_total to all reachable vertices.
    # Due to problem guarantees, this includes all vertices in V2 = {N1+1, ..., N_total}.
    distN = bfs(N_total, N_total, adj)
    
    # Find the maximum distance from vertex N_total to any vertex in V2.
    # This value is denoted as D2. Initialize it to 0.
    D2 = 0
    # Iterate through vertices from N1+1 to N_total.
    for i in range(N1 + 1, N_total + 1):
         # Problem guarantees ensure V2 is connected and contains vertex N_total,
         # so all vertices i in {N1+1, ..., N_total} are reachable from N_total. distN[i] will be non-negative.
         # Update D2 with the maximum distance found so far.
         # Add a safety check for reachability.
        if distN[i] != -1: 
           D2 = max(D2, distN[i])
        # else: pass # This case should not happen based on problem statement guarantees.

    # The problem asks for the maximum possible shortest path distance 'd' between vertex 1 and vertex N_total
    # after adding exactly one edge (u, v) where u is in V1 and v is in V2.
    # The shortest path in the modified graph will be of the form 1 -> ... -> u -> v -> ... -> N_total.
    # Its length is d(1, u) + 1 + d(v, N_total).
    # To maximize this length, we should choose u maximizing d(1, u) and v maximizing d(v, N_total) = d(N_total, v).
    # The maximum possible shortest path length is thus max_{u in V1} d(1,u) + 1 + max_{v in V2} d(N_total,v) = D1 + 1 + D2.
    # The '+1' accounts for the newly added edge (u, v).
    # Print the final calculated result to standard output.
    print(D1 + D2 + 1)

# Call the main solve function to execute the program logic when the script is run.
solve()