import collections

class Solution:
  def minMaxWeight(self, n: int, edges: list[list[int]], threshold: int) -> int:
    
    # check(max_allowed_weight) function:
    # Determines if a valid subgraph can be formed using only edges with weights
    # less than or equal to max_allowed_weight.
    def check(max_allowed_weight: int) -> bool:
      # adj_rev[X] will store a list of Y such that X -> Y is an edge in G_rev.
      # G_rev is the graph formed by taking original edges with weight <= max_allowed_weight,
      # and reversing their directions.
      adj_rev = [[] for _ in range(n)]
      for u_orig, v_orig, w_edge in edges:
        if w_edge <= max_allowed_weight:
          # Original edge: u_orig -> v_orig
          # Reversed edge in G_rev: v_orig -> u_orig
          adj_rev[v_orig].append(u_orig)
      
      # Perform BFS starting from node 0 in G_rev.
      # Condition: Node 0 must be reachable from all other nodes in the original selected subgraph.
      # This is equivalent to: Node 0 can reach all other nodes in G_rev.
      q = collections.deque()
      q.append(0) 
      visited = {0}
      nodes_reached_count = 1
      
      # n >= 2 from constraints. So nodes_reached_count must become n.
      # If n=1, it would be true here, but constraints say n >= 2.

      while q:
        curr_node = q.popleft()
        
        for neighbor in adj_rev[curr_node]: # Edge: curr_node -> neighbor in G_rev
          if neighbor not in visited:
            visited.add(neighbor)
            q.append(neighbor)
            nodes_reached_count += 1
            # Optimization: if all nodes are already reached, no need to continue BFS.
            if nodes_reached_count == n:
                return True 
            
      return nodes_reached_count == n

    # Determine search range for binary search on max_allowed_weight
    # Smallest possible edge weight is 1. Largest is max weight in edges (up to 10^6).
    
    # If there are no edges, and n > 1, it's impossible to satisfy reachability.
    # n >= 2 is a constraint, so n > 1 is always true.
    if not edges:
        return -1

    # Find the maximum weight present in the graph to set the upper bound for binary search.
    # Using 10^6 (max possible weight from constraints) is also fine for `high_bound`.
    max_w_in_graph = 0
    for _, _, w_val in edges:
        max_w_in_graph = max(max_w_in_graph, w_val)
    
    ans = -1
    low_bound = 1          # Smallest possible edge weight (constraints: W_i >= 1)
    high_bound = max_w_in_graph 

    while low_bound <= high_bound:
      mid_weight = low_bound + (high_bound - low_bound) // 2
      
      if check(mid_weight):
        ans = mid_weight
        high_bound = mid_weight - 1 # Try to find an even smaller max_allowed_weight
      else:
        low_bound = mid_weight + 1  # max_allowed_weight is too small, need to increase it
        
    return ans