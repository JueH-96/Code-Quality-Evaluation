from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # The first element is always included in the cost
        cost = nums[0]
        # We need to find the two smallest elements from the rest of the array
        smallest_two = sorted(nums[1:3])
        cost += sum(smallest_two)
        # Iterate through the rest of the array to find the two smallest elements
        for num in nums[3:]:
            if num < smallest_two[1]:
                smallest_two[1] = num
                smallest_two.sort()
        return cost + smallest_two[1]