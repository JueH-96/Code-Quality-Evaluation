import collections
from typing import List

class Solution:
  def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
    # adj will store the graph. adj[u] is a list of nodes v such that there's an edge u->v.
    # Using lists of lists for adjacency list representation.
    adj = [[] for _ in range(n)]
    
    # Add initial roads: i to i+1 for all 0 <= i < n-1.
    # These roads form the baseline path 0 -> 1 -> ... -> n-1.
    # Example: n=5, initial roads are (0,1), (1,2), (2,3), (3,4).
    for i in range(n - 1):
        adj[i].append(i + 1)
        
    results = [] # This list will store the shortest path length after each query.
    
    # Process each query one by one
    for u_query, v_query in queries:
        # Add the new unidirectional road from u_query to v_query to the graph.
        # The graph's structure is modified and this change persists for subsequent queries.
        adj[u_query].append(v_query)
        
        # Perform Breadth-First Search (BFS) to find the shortest path 
        # from city 0 to city n-1. Since all road lengths (edge weights) are 1,
        # BFS is guaranteed to find the shortest path in terms of number of roads.
        
        # distances[i] will store the length of the shortest path from node 0 to node i.
        # Initialize all distances to positive infinity.
        distances = [float('inf')] * n
        # The distance from the start node (city 0) to itself is 0.
        distances[0] = 0
        
        # Queue for BFS. It will store nodes to visit.
        # We use collections.deque for efficient append (enqueue) and popleft (dequeue) operations.
        queue = collections.deque([0]) # Start BFS from city 0.
        
        # This variable will store the length of the shortest path to the target city (n-1).
        # It's initialized to infinity. If n-1 is reached, this will be updated.
        # The problem implies n-1 is always reachable (e.g., via the initial 0->1->...->n-1 path).
        path_len_to_target = float('inf')

        # BFS loop continues as long as there are nodes to visit in the queue.
        while queue:
            curr_node = queue.popleft() # Get the next node to process.
            
            # The shortest distance from city 0 to curr_node found so far.
            curr_dist = distances[curr_node]

            # Check if the current node is the target city (n-1).
            if curr_node == n - 1:
                # We have found a path to the target city.
                # Since BFS explores nodes layer by layer (i.e., nodes closer to the start
                # are visited before nodes farther away), the first time we reach
                # the target node, it's guaranteed to be via a shortest path.
                path_len_to_target = curr_dist
                # We can stop the BFS now as we've found the shortest path to n-1.
                # Any other paths to n-1 still in the queue or reachable later
                # cannot be shorter.
                break 
            
            # Explore all neighbors of curr_node.
            # A neighbor is a city reachable by a direct road from curr_node.
            for neighbor in adj[curr_node]:
                # If this neighbor has not been visited yet (its distance is still infinity),
                # it means this is the first time we are reaching it.
                # In BFS for unweighted graphs, the first time a node is reached is via its shortest path.
                if distances[neighbor] == float('inf'): 
                    # Update its distance: distance to neighbor is distance to curr_node + 1.
                    distances[neighbor] = curr_dist + 1
                    # Add the neighbor to the queue to visit its neighbors later.
                    queue.append(neighbor)
        
        # After the BFS, path_len_to_target holds the shortest distance from city 0 to city n-1.
        # (Alternatively, distances[n-1] would also hold this value if the 'break' condition was met).
        results.append(path_len_to_target)
            
    return results