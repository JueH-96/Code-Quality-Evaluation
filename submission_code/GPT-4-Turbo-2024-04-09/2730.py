class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        import heapq
        
        # Priority queue to maximize the effect of doubling
        # We use negative values because heapq is a min-heap by default
        max_heap = []
        for num in nums:
            # Push the current number and the potential gain from doubling it
            heapq.heappush(max_heap, (-num, num))
        
        # Perform k operations
        for _ in range(k):
            # Get the most beneficial number to double
            neg_num, original_num = heapq.heappop(max_heap)
            doubled_num = original_num * 2
            
            # Push the doubled number into the heap
            heapq.heappush(max_heap, (-doubled_num, doubled_num))
        
        # Calculate the final OR result
        result = 0
        while max_heap:
            _, num = heapq.heappop(max_heap)
            result |= num
        
        return result