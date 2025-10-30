class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        from collections import defaultdict
        import heapq

        # Create an adjacency list to count the degree of each node
        degree = defaultdict(int)
        for u, v, w in edges:
            degree[u] += 1
            degree[v] += 1

        # Create a max-heap based on edge weights
        max_heap = []
        for u, v, w in edges:
            heapq.heappush(max_heap, (-w, u, v))

        # Initialize the sum of weights
        max_sum = 0

        # Keep track of the degree of each node after removals
        current_degree = defaultdict(int)

        # Process the edges in descending order of weights
        while max_heap:
            w, u, v = heapq.heappop(max_heap)
            w = -w  # Convert back to positive weight

            # Check if adding this edge violates the degree constraint
            if current_degree[u] < k and current_degree[v] < k:
                # Add this edge to the result
                max_sum += w
                current_degree[u] += 1
                current_degree[v] += 1

        return max_sum