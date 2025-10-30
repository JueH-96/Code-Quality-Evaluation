# YOUR CODE HERE
import sys
# Using sys.stdin.readline for faster input reading compared to input()
from collections import deque

def solve():
    # Read the number of cities
    N = int(sys.stdin.readline())
    
    # Handle the edge case where there is only one city.
    # No travel is needed, so the distance is 0.
    if N == 1:
      print(0)
      return

    # Adjacency list representation of the graph (tree).
    # Keys are node IDs (1 to N), values are lists of tuples (neighbor, weight).
    adj = {}
    # Variable to store the sum of all edge weights (lengths).
    total_edge_weight = 0

    # Read N-1 edges from input
    for _ in range(N - 1):
        # Read u, v (cities connected) and c (length of road)
        # Use sys.stdin.readline().split() and map to int for efficiency
        line = sys.stdin.readline().split()
        u = int(line[0])
        v = int(line[1])
        c = int(line[2])
        
        # Ensure nodes u and v exist as keys in the adjacency list before appending
        # This ensures that even nodes initially appearing only as 'v' are added.
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        
        # Add edge to adjacency list for both directions since roads are bidirectional
        adj[u].append((v, c))
        adj[v].append((u, c))
        
        # Accumulate the total weight of all edges
        total_edge_weight += c

    # Define a Breadth-First Search (BFS) function to find the farthest node 
    # from a given start node in the tree and the distance to it.
    # This is used to find the diameter of the tree.
    def bfs(start_node):
        # Initialize distances array. Use -1 to indicate node not visited.
        # Array size N+1 allows using 1-based indexing (indices 1 to N). Index 0 is unused.
        distances = [-1] * (N + 1)
        # Distance from start node to itself is 0.
        distances[start_node] = 0
        
        # Initialize a queue for BFS with the start node and its distance (0).
        # collections.deque offers efficient appends and pops from both ends.
        queue = deque([(start_node, 0)])
        
        # Variables to keep track of the maximum distance found so far and the corresponding node ID.
        max_dist = 0
        farthest_node = start_node
        
        # Standard BFS loop: continues as long as the queue is not empty.
        while queue:
            # Dequeue the node 'u' and its distance 'dist_u' from the start node
            u, dist_u = queue.popleft()
            
            # Check if this node 'u' is farther than the current farthest node found.
            # If yes, update max_dist and farthest_node.
            if dist_u > max_dist:
                max_dist = dist_u
                farthest_node = u
                
            # Explore neighbors of the current node u
            # Check if u has entries in the adjacency list. This is guaranteed for connected components
            # in a tree with N>=2, but it's a safe check.
            if u in adj: 
                 for v, w in adj[u]: # v is neighbor, w is weight of edge (u, v)
                    # If neighbor v has not been visited yet (its distance is still -1)
                    if distances[v] == -1: 
                        # Mark v as visited by setting its distance
                        distances[v] = dist_u + w
                        # Enqueue neighbor v along with its calculated distance from the start node
                        queue.append((v, distances[v]))

        # After BFS completes, return the ID of the farthest node and its distance
        return farthest_node, max_dist

    # The diameter of a tree can be found using two BFS traversals:
    # Step 1: Start BFS from an arbitrary node (e.g., node 1). Node 1 is guaranteed to exist for N >= 1.
    # Find the node `endpoint1` that is farthest away from node 1.
    # We only need the node ID `endpoint1` for the next step, the distance is not used here.
    endpoint1, _ = bfs(1)  
    
    # Step 2: Start BFS from the node `endpoint1` found in Step 1.
    # Find the node `endpoint2` that is farthest away from `endpoint1`.
    # The distance `diameter` to `endpoint2` is the diameter of the tree.
    # Note: `endpoint2` is the other endpoint of a diameter path starting at `endpoint1`.
    _, diameter = bfs(endpoint1) # We need the maximum distance found, which is the diameter.

    # The problem asks for the minimum travel distance to visit all cities starting from any city.
    # This minimum distance is known to be equal to 2 * (sum of all edge weights) - (diameter of the tree).
    # Let L be the total edge weight sum. The formula is 2L - D.
    min_travel_distance = 2 * total_edge_weight - diameter
    
    # Print the calculated minimum travel distance.
    print(min_travel_distance)

# Call the solve function to execute the entire logic.
solve()