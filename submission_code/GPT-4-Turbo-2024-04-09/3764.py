class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        import heapq
        
        # Create a max-heap to store the largest values from each row considering their limits
        max_heap = []
        
        # Iterate over each row in the grid
        for row, limit in zip(grid, limits):
            # Sort the row in descending order to get the largest elements first
            sorted_row = sorted(row, reverse=True)
            # Take the top 'limit' elements from the sorted row
            for value in sorted_row[:limit]:
                # Push the value into the max-heap
                heapq.heappush(max_heap, -value)
        
        # We need to take exactly k elements, so we sum the top k largest elements from the max-heap
        max_sum = 0
        for _ in range(k):
            max_sum += -heapq.heappop(max_heap)
        
        return max_sum