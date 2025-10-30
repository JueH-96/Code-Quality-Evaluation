class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        """
        We have a tree of n nodes (n = len(edges) + 1) and n-1 edges.
        Each edge has a weight. We want to remove zero or more edges
        so that no node exceeds degree k, and the sum of the remaining
        edges' weights is maximized.

        Key Idea:
        - Since we only care about limiting degrees and maximizing the sum,
          we can sort the edges by descending weight.
        - Then, greedily pick edges from largest to smallest, adding each
          edge as long as neither of its endpoints would exceed degree k
          by including that edge.
        - This ensures we keep as many of the large-weight edges as possible
          while respecting the degree constraint.

        Time complexity is dominated by sorting the edges, O(n log n),
        which is acceptable for up to 10^5 edges.
        """

        n = len(edges) + 1

        # Sort edges in descending order by weight
        edges.sort(key=lambda x: x[2], reverse=True)

        # Track degrees of each node
        degree = [0] * n

        total_weight = 0
        for u, v, w in edges:
            # If both endpoints still have room (degree < k), keep this edge
            if degree[u] < k and degree[v] < k:
                degree[u] += 1
                degree[v] += 1
                total_weight += w

        return total_weight