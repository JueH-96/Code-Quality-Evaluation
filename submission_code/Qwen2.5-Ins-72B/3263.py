from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # The first element is always part of the cost
        first_cost = nums[0]
        
        # Find the two smallest elements in the rest of the array
        min1, min2 = float('inf'), float('inf')
        for num in nums[1:]:
            if num < min1:
                min2 = min1
                min1 = num
            elif num < min2:
                min2 = num
        
        # The total cost is the sum of the first element and the two smallest elements
        return first_cost + min1 + min2