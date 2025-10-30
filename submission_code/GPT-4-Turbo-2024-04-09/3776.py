class Solution:
    def minCost(self, nums: List[int]) -> int:
        from heapq import heappop, heappush, heapify
        
        # If the list is very short, handle it directly
        if len(nums) <= 2:
            return max(nums)
        
        # Use a min-heap to always access the smallest elements
        heap = []
        heapify(heap)
        for num in nums:
            heappush(heap, num)
        
        total_cost = 0
        
        # While more than two elements remain, process the smallest two
        while len(heap) > 2:
            first = heappop(heap)
            second = heappop(heap)
            cost = max(first, second)
            total_cost += cost
            # Since we can only remove two elements at a time, we don't push anything back
        
        # Handle the last one or two elements
        if len(heap) == 2:
            total_cost += max(heappop(heap), heappop(heap))
        elif len(heap) == 1:
            total_cost += heappop(heap)
        
        return total_cost