import collections
from typing import List

class Solution:
  """
  Finds the minimum possible maximum edge weight in a subgraph satisfying reachability and out-degree constraints using binary search and BFS.

  The problem asks for the minimum possible value of the maximum edge weight in a subgraph obtained by removing edges from the original graph. This subgraph must satisfy three conditions:
  1. Node 0 must be reachable from all other nodes (1 to n-1).
  2. Each node must have at most `threshold` outgoing edges.
  3. The subgraph's edges are a subset of the original graph's edges.

  The core idea is to use binary search on the possible values of the maximum edge weight. Let this maximum weight be `W_max`. For a given `W_max`, we need to check if it's possible to select a subset of edges from the original graph such that:
  - All selected edges have weights less than or equal to `W_max`.
  - The conditions (1) and (2) are met using only the selected edges.

  The check for a given `W_max` can be efficiently performed using a graph traversal algorithm on a modified graph. Instead of checking reachability towards node 0 directly, we can reverse all edges and check reachability from node 0 outwards.
  Let `G` be the original graph. Let `G_filtered` be the subgraph containing only edges from `G` with weights `<= W_max`.
  Let `G_rev_filtered` be the graph obtained by reversing all edges in `G_filtered`.
  
  The original condition "Node 0 must be reachable from all other nodes" using selected edges in `G` is equivalent to "All other nodes must be reachable from Node 0" using the corresponding selected reversed edges in `G_rev_filtered`.
  The original constraint "Each node `u` has at most `threshold` outgoing edges" using selected edges in `G` is equivalent to "Each node `u` has at most `threshold` incoming edges" using the corresponding selected reversed edges in `G_rev_filtered`.

  We can perform a Breadth-First Search (BFS) starting from node 0 in `G_rev_filtered`. During the BFS, we keep track of the number of incoming edges selected for each node (`in_degree_count`). An edge `u -> v` in `G_rev_filtered` is considered traversable only if `in_degree_count[v]` is currently less than `threshold`. If we traverse this edge, we increment `in_degree_count[v]`. If the BFS starting from node 0 manages to visit all `n` nodes, it means a valid subgraph exists for the given `W_max`.

  The range for binary search is [0, 10^6], as edge weights are between 1 and 10^6. We search for the minimum `W_max` for which the check function returns True.

  Args:
    n: The number of nodes in the graph (0 to n-1).
    edges: A list of edges, where each edge is `[u, v, w]`, representing a directed edge from `u` to `v` with weight `w`.
    threshold: The maximum allowed number of outgoing edges for any node in the resulting subgraph.

  Returns:
    The minimum possible value of the maximum edge weight in a valid subgraph. Returns -1 if it's impossible to satisfy the conditions.
  """
  def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
    
    # Check function: Determines if it's possible to satisfy the conditions
    # with a maximum allowed edge weight of W_max.
    def check(W_max: int) -> bool:
      """
      Checks if a valid subgraph exists where all edge weights are <= W_max,
      node 0 is reachable from all other nodes, and each node has out-degree <= threshold.

      Args:
        W_max: The maximum allowed edge weight.

      Returns:
        True if such a subgraph exists, False otherwise.
      """
      # Build the reversed graph containing only edges with weight <= W_max.
      # graph_rev[v] contains list of (u, w) meaning reversed edge v -> u exists (original edge u -> v).
      # This structure represents adjacency list for the reversed graph.
      graph_rev = collections.defaultdict(list)
      for u, v, w in edges:
          if w <= W_max:
              # Add edge v -> u with weight w to the reversed graph representation
              graph_rev[v].append((u, w)) 

      # Perform BFS starting from node 0 on the reversed graph.
      # Constraint: node `v`'s out-degree in original graph <= threshold.
      # Translated: node `v`'s in-degree in reversed graph <= threshold for selected edges.
      
      # in_degree_count[v]: tracks selected incoming edges to node `v` in reversed graph.
      # This count corresponds to the out-degree of node `v` in the original graph using selected edges.
      in_degree_count = [0] * n 
      
      # Initialize BFS queue with starting node 0.
      q = collections.deque([0])
      # Keep track of visited nodes. Start with node 0 marked as visited.
      visited = {0}
      
      # Perform BFS traversal
      while q:
          # Get the current node `u` from the front of the queue.
          u = q.popleft() 

          # Explore neighbors `v` reachable from `u` via edge `u -> v` in the reversed graph.
          # `(v, w)` here means there's an edge `u -> v` with weight `w` in the reversed graph.
          for v, w in graph_rev[u]:
              # Check the constraint on node `v` (the destination of the edge `u -> v` in the reversed graph).
              # `v`'s out-degree constraint in the original graph is checked via its in-degree count in the reversed graph.
              if in_degree_count[v] < threshold:
                  # If the constraint permits selecting this edge:
                  # Increment the in-degree count for `v`. This effectively marks the edge as 'selected'
                  # for the purpose of path finding and constraint tracking.
                  in_degree_count[v] += 1 
                  
                  # If neighbor `v` hasn't been visited yet during this BFS:
                  if v not in visited:
                      # Mark `v` as visited.
                      visited.add(v)
                      # Add `v` to the queue to explore its neighbors later.
                      q.append(v)
      
      # After the BFS completes, check if all `n` nodes were visited.
      # If `len(visited) == n`, it means node 0 can reach all other nodes in the reversed graph
      # while respecting the constraints. This implies that in the original graph, all nodes
      # can reach node 0 using the corresponding selected edges, satisfying all conditions.
      return len(visited) == n

    # Binary search for the minimum possible maximum edge weight.
    # The search space for the maximum weight is [0, 10^6].
    low = 0
    # The upper bound is set to 10^6 + 1 to use the standard binary search range `[low, high)`.
    high = 10**6 + 1  
    ans = -1 # Initialize answer to -1, representing the case where it's impossible.

    # Perform binary search. The loop continues as long as the search range `[low, high)` is valid.
    while low < high:
        # Calculate the middle value. Using this formula avoids potential integer overflow.
        mid = low + (high - low) // 2
        
        # Check if a valid subgraph exists with maximum edge weight `mid`.
        if check(mid):
            # If `check(mid)` is true, it means `mid` is a feasible maximum weight.
            # This `mid` could be the minimum possible value, so we store it as a potential answer.
            ans = mid
            # Try to find an even smaller valid maximum weight by searching in the lower half: `[low, mid)`.
            high = mid 
        else:
            # If `check(mid)` is false, `mid` is too small to satisfy the conditions.
            # We must consider larger maximum weights. Search in the upper half: `[mid + 1, high)`.
            low = mid + 1

    # The loop terminates when low == high. `ans` holds the minimum `mid` value 
    # for which `check(mid)` returned true. If no such `mid` was found (check never returned true),
    # `ans` remains at its initial value of -1.
    return ans