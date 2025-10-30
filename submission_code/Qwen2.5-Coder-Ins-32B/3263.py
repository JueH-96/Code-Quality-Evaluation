class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # The first element is always part of the cost
        first_element = nums[0]
        
        # We need to find the two smallest elements after the first element
        # to minimize the cost of the other two subarrays
        min1, min2 = float('inf'), float('inf')
        
        for num in nums[1:]:
            if num < min1:
                min2 = min1
                min1 = num
            elif num < min2:
                min2 = num
        
        return first_element + min1 + min2