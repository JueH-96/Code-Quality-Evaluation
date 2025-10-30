import sys
from typing import List
from collections import defaultdict

# Attempt to increase the recursion depth limit for potentially deep trees.
# Python's default limit (often 1000) might be insufficient for N=10^5.
# This might not work on all platforms or execution environments.
try:
    # Set recursion depth to accommodate maximum possible tree depth N=10^5, plus some buffer.
    # E.g., 110000 for N=10^5.
    sys.setrecursionlimit(110000) 
except Exception as e:
    # If setting the limit fails (e.g., due to OS restrictions), print a warning.
    # The code might still pass if test cases don't involve extremely deep trees.
    print(f"Warning: Failed to set recursion depth: {e}", file=sys.stderr) 

class Solution:
  """
  Solves the problem of finding the time for all nodes to be marked in a tree,
  for each possible starting node. This implementation uses a dynamic programming 
  approach on trees. It involves two Depth First Search (DFS) passes to compute 
  the necessary path lengths (eccentricity) for each node, considering the custom 
  edge weights based on node parity. The overall time complexity is O(N), 
  where N is the number of nodes, after building the adjacency list.
  """
  def timeTaken(self, edges: List[List[int]]) -> List[int]:
      """
      Calculates the maximum time required to mark all nodes in the tree, 
      assuming node i is the initially marked node at time 0, for each i from 0 to n-1.
      
      Args:
          edges: A list of edge pairs [u, v], representing the undirected edges of the tree.
                 Example: [[0,1], [0,2]] for a tree with 3 nodes.
      
      Returns:
          A list `times` where `times[i]` is the time until all nodes are marked, 
          if node i is the starting node.
      """
      n = len(edges) + 1 # Number of nodes in the tree
      # Constraints state N >= 2, so n >= 2. No need for n=0 or n=1 checks.
      
      # Build adjacency list representation of the tree
      adj = defaultdict(list)
      for u, v in edges:
          adj[u].append(v)
          adj[v].append(u)

      # DP arrays to store the maximum distances/times.
      # D_down[u]: max time from u downwards to any node in its subtree.
      # D_up[u]: max time from u upwards/sideways to any node outside its subtree.
      # Initialize with -1 to indicate 'not computed yet'. This isn't strictly necessary
      # for correctness if DFS calls are ordered properly, but can be useful for debugging.
      D_down = [-1] * n 
      D_up = [-1] * n
      
      # Helper function to calculate the time cost associated with marking node v from neighbor u.
      # The cost depends on the parity of the destination node v.
      def get_weight(u, v):
          """ Returns the time cost to mark node v when triggered by node u. """
          # If node v is odd, it gets marked at time t+1 if neighbor u was marked at time t. Cost = 1.
          # If node v is even, it gets marked at time t+2 if neighbor u was marked at time t. Cost = 2.
          return 1 if v % 2 == 1 else 2

      # First DFS pass (post-order traversal logic):
      # Computes D_down[u] for all nodes u.
      def dfs1(u, p):
          """ Performs the first DFS pass to compute D_down values. """
          max_dist = 0 # Initialize max distance downwards from u to 0.
          # Iterate through neighbors of u
          for v in adj[u]:
              if v != p: # Ensure v is a child node (not the parent p)
                  # Recursively call dfs1 for child v first. This ensures post-order calculation.
                  dfs1(v, u) 
                  # Update max_dist: it's the maximum over all children of 
                  # (cost to reach child v + max distance from v downwards).
                  max_dist = max(max_dist, get_weight(u, v) + D_down[v])
          
          # Store the computed max downward distance for node u.
          D_down[u] = max_dist

      # Second DFS pass (pre-order traversal logic):
      # Computes D_up[v] for all children v of u, using D_up[u] and sibling information.
      def dfs2(u, p):
          """ Performs the second DFS pass to compute D_up values. """
          children = [] # List of children nodes of u
          child_vals = [] # List of corresponding maximum downward path values starting from u through each child
          # Collect children and precompute their max downward path values starting from u
          for v in adj[u]:
              if v != p: # If v is a child
                  children.append(v)
                  # This value represents max distance: u -> v -> ... (deepest in v's subtree)
                  child_vals.append(get_weight(u, v) + D_down[v])

          num_children = len(children)
          if num_children == 0: return # Base case: u is a leaf in this traversal direction

          # Optimization: Calculate prefix and suffix maximums of child_vals
          # This allows efficient O(1) lookup for the maximum value among siblings of a node.
          prefix_max = [float('-inf')] * num_children
          suffix_max = [float('-inf')] * num_children
          
          current_max = float('-inf')
          for i in range(num_children):
              current_max = max(current_max, child_vals[i])
              prefix_max[i] = current_max

          current_max = float('-inf')
          for i in range(num_children - 1, -1, -1):
              current_max = max(current_max, child_vals[i])
              suffix_max[i] = current_max

          # Compute D_up for each child v and recurse down the tree
          for i in range(num_children):
              v = children[i] # Current child node
              
              # Find the maximum path value starting from parent u, but NOT going towards child v.
              # This path can either go further up (captured by D_up[u]) or into a sibling's subtree.
              max_sibling_val = float('-inf')
              if i > 0: # Check max among siblings to the left
                  max_sibling_val = max(max_sibling_val, prefix_max[i-1])
              if i < num_children - 1: # Check max among siblings to the right
                  max_sibling_val = max(max_sibling_val, suffix_max[i+1])
              
              # The maximum distance path starting from parent u, excluding the path towards v,
              # is the maximum of the path going upwards (D_up[u]) and the maximum path going
              # into any sibling's subtree (max_sibling_val).
              max_val_from_parent = max(D_up[u], max_sibling_val)
              # Note: Python's max function handles float('-inf') correctly.

              # Calculate D_up[v]: This is the cost to reach parent u from v, plus the maximum
              # distance reachable from u without going back down towards v.
              # The cost get_weight(v, u) depends on the destination node u's parity.
              D_up[v] = get_weight(v, u) + max_val_from_parent
              
              # Recursively call dfs2 for the child v to compute D_up for its children.
              dfs2(v, u)

      # ----- Main Execution Logic -----
      
      # Choose an arbitrary root, e.g., node 0. Run DFS1 from the root.
      # This computes D_down values for all nodes in the tree.
      dfs1(0, -1) 

      # Initialize D_up for the root node. Max distance upwards from root is 0.
      D_up[0] = 0 
      
      # Run DFS2 from the root node. This computes D_up values for all other nodes.
      dfs2(0, -1)

      # The final answer for each node i is its eccentricity in this weighted directed graph sense.
      # Eccentricity(i) = max distance from i to any other node j.
      # This is calculated as max(D_down[i], D_up[i]).
      times = [0] * n
      for i in range(n):
          times[i] = max(D_down[i], D_up[i])

      return times