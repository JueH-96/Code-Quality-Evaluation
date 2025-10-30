from typing import List
import heapq

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        # We will use a max heap where we prioritize the elements that can contribute the most to the OR
        # by doubling them. The contribution is measured by the new bits they can set.

        # Initialize the heap with tuples of (-potential_gain, operations_left, current_value, index)
        heap = []
        for i, num in enumerate(nums):
            if k > 0:
                next_val = num * 2
                gain = next_val - num
                heapq.heappush(heap, (-gain, 1, next_val, i))
        
        while k > 0 and heap:
            gain, ops, new_val, idx = heapq.heappop(heap)
            gain = -gain
            # Update the number
            nums[idx] = new_val
            k -= 1
            # Push the next possible operation on the same number
            if k - ops >= 0:
                next_val = new_val * 2
                next_gain = next_val - new_val
                heapq.heappush(heap, (-next_gain, ops + 1, next_val, idx))
        
        # Compute the final OR
        result = 0
        for num in nums:
            result |= num
        return result