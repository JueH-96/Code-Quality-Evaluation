import math
from typing import List

class Solution:
    """
    Solves the shift distance problem using the Floyd-Warshall algorithm
    to precompute all-pairs shortest paths between letters ('a' through 'z').
    The alphabet operations form a cycle graph where nodes are letters and
    edges represent single shifts (next or previous letter) with associated costs.
    """
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        """
        Calculates the minimum total cost to transform string s into string t
        by applying character shifts.

        Args:
            s: The starting string.
            t: The target string. Both s and t have the same length and consist of lowercase English letters.
            nextCost: A list of length 26 where nextCost[i] is the cost to shift the i-th letter
                      (0='a', ..., 25='z') to the next letter in the alphabet (wrapping 'z' to 'a').
            previousCost: A list of length 26 where previousCost[i] is the cost to shift the i-th letter
                          to the previous letter in the alphabet (wrapping 'a' to 'z'). Costs are non-negative integers.
        
        Returns:
            The minimum total cost (shift distance) required to transform s into t. The total cost can be large, but will fit in standard integer types.
        """
        
        # Use float('inf') for representing infinity in shortest path calculation.
        # This is a standard practice and Python handles arithmetic with float('inf') correctly.
        infinity = float('inf') 
        
        N = 26 # Number of letters in the English alphabet

        # Initialize the distance matrix `dist`. dist[i][j] will store the minimum cost
        # to transform the character represented by index i (0='a', ..., 25='z') 
        # to the character represented by index j.
        # Initialize all distances to infinity.
        dist = [[infinity] * N for _ in range(N)]
        
        # The cost to transform a character to itself is 0. Initialize the diagonal elements.
        for i in range(N):
            dist[i][i] = 0
        
        # Initialize the costs for direct one-step shifts based on the input costs.
        # This setup represents the initial graph edges.
        # For each character `i`, there's an edge to the next character `(i+1)%N`
        # and an edge to the previous character `(i-1+N)%N`.
        for i in range(N):
            next_node = (i + 1) % N  # Index of the next letter in the alphabet cycle
            prev_node = (i - 1 + N) % N # Index of the previous letter in the alphabet cycle
            
            # The cost to move from character `i` to `next_node` is `nextCost[i]`.
            # Update dist[i][next_node] only if this direct path is cheaper than any existing path 
            # (relevant if N were small, e.g., N=2, where next == prev). Using min is robust.
            dist[i][next_node] = min(dist[i][next_node], nextCost[i])
            
            # The cost to move from character `i` to `prev_node` is `previousCost[i]`.
            dist[i][prev_node] = min(dist[i][prev_node], previousCost[i])

        # Apply the Floyd-Warshall algorithm to compute all-pairs shortest paths.
        # This algorithm iterates through all possible intermediate nodes `k`
        # to potentially find shorter paths between any pair of nodes `(i, j)`.
        for k in range(N): # Consider `k` as an intermediate node
            for i in range(N): # Consider `i` as the start node
                for j in range(N): # Consider `j` as the end node
                    # Check if paths from `i` to `k` and from `k` to `j` exist (i.e., are not infinity).
                    # This check prevents issues with infinity arithmetic if a path doesn't exist.
                    if dist[i][k] != infinity and dist[k][j] != infinity:
                        # Calculate the cost of the path passing through `k`.
                        cost_via_k = dist[i][k] + dist[k][j]
                        # If the path through `k` is shorter than the currently known shortest path from `i` to `j`,
                        # update the shortest path distance.
                        if cost_via_k < dist[i][j]:
                           dist[i][j] = cost_via_k

        # Calculate the total shift distance for transforming string s to string t.
        # Since operations on characters at different indices are independent,
        # the total cost is the sum of minimum costs for transforming each s[i] to t[i].
        total_cost = 0
        # Iterate through the strings character by character.
        for i in range(len(s)):
            # Convert characters s[i] and t[i] to their corresponding 0-based indices (0 for 'a', ..., 25 for 'z').
            s_idx = ord(s[i]) - ord('a')
            t_idx = ord(t[i]) - ord('a')
            
            # Retrieve the precomputed minimum cost to transform character s_idx to t_idx from the distance matrix.
            cost_for_pair = dist[s_idx][t_idx]
            
            # The problem implies that transformation is always possible. Due to the cyclic nature of the alphabet graph
            # and non-negative edge weights, all nodes are reachable from all other nodes.
            # Therefore, cost_for_pair should always be finite (not infinity).
            # If cost_for_pair were infinity, it would indicate an unreachable state, potentially due to an error or unexpected input.
            # We assume reachability as per the problem statement and graph structure.

            # Add the minimum cost for this character pair to the total cost.
            total_cost += cost_for_pair
            
        # Return the computed total minimum cost. Since all input costs are integers, 
        # the shortest path costs and the total cost will also be integers.
        # Python's integers handle arbitrary size, so large costs are managed correctly.
        # No explicit type casting is needed unless required by specific platform constraints.
        # The function signature implies integer return type is expected.
        return int(total_cost) # Casting to int ensures integer return type, although total_cost should already be integer if inputs were.