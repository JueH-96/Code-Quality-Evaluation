class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        
        # The problem of finding the minimum cost to transform one character to another
        # can be modeled as finding the shortest path on a graph. The 26 lowercase
        # English letters form the vertices of this graph.

        # For each character `c` (represented by its 0-indexed position `i` in the alphabet),
        # there are two possible single-step operations:
        # 1. Shift to the next character: This is a directed edge from vertex `i` to
        #    `(i + 1) % 26` with a weight of `nextCost[i]`.
        # 2. Shift to the previous character: This is a directed edge from vertex `i` to
        #    `(i - 1 + 26) % 26` with a weight of `previousCost[i]`.
        
        # Since we need to find the minimum transformation cost for various pairs of
        # characters (from string `s` to `t`), it's efficient to pre-compute the
        # shortest path costs between all pairs of letters. The Floyd-Warshall algorithm
        # is a standard method for solving the All-Pairs Shortest Path (APSP) problem.
        
        N = 26
        infinity = float('inf')
        
        # `dist[i][j]` will store the minimum cost to transform character `i` to `j`.
        dist = [[infinity] * N for _ in range(N)]
        
        # Initialize the distance matrix. The cost to transform a character to itself is 0.
        for i in range(N):
            dist[i][i] = 0
            
        # Set the weights for direct edges (single-step shifts).
        for i in range(N):
            next_char_idx = (i + 1) % N
            # Using min is robust, though simple assignment works here since edges
            # from a node `i` to its next and previous nodes are distinct.
            dist[i][next_char_idx] = min(dist[i][next_char_idx], nextCost[i])
            
            prev_char_idx = (i - 1 + N) % N
            dist[i][prev_char_idx] = min(dist[i][prev_char_idx], previousCost[i])

        # Run the Floyd-Warshall algorithm.
        # It iteratively considers each vertex `k` as an intermediate point in paths.
        # The complexity is O(N^3), which is efficient for N=26.
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    # If the path from `i` to `j` via `k` is shorter, update the distance.
                    if dist[i][k] != infinity and dist[k][j] != infinity:
                        path_via_k = dist[i][k] + dist[k][j]
                        if path_via_k < dist[i][j]:
                            dist[i][j] = path_via_k
                            
        # With the `dist` matrix fully computed, calculate the total shift distance.
        # The total cost is the sum of minimum costs for transforming each s[i] to t[i].
        total_cost = 0
        for i in range(len(s)):
            s_idx = ord(s[i]) - ord('a')
            t_idx = ord(t[i]) - ord('a')
            
            # Add the precomputed shortest path cost for this character pair.
            total_cost += dist[s_idx][t_idx]
            
        return total_cost