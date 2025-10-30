from typing import List
import heapq

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Create a max heap from the negative values of nums
        max_heap = [-num for num in nums]
        heapq.heapify(max_heap)
        
        current_sum = sum(nums)
        operations = 0
        
        # If the current sum is already less than the target, it's impossible
        if current_sum < target:
            return -1
        
        while current_sum > target:
            largest = -heapq.heappop(max_heap)
            current_sum -= largest
            half = largest // 2
            heapq.heappush(max_heap, -half)
            heapq.heappush(max_heap, -half)
            current_sum += half * 2
            operations += 1
        
        # Now we need to check if we can form the target sum
        current_sum = 0
        while max_heap and current_sum < target:
            largest = -heapq.heappop(max_heap)
            if current_sum + largest <= target:
                current_sum += largest
        
        if current_sum == target:
            return operations
        else:
            return -1