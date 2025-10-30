from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        """
        Calculates the minimum cost to divide the array into three disjoint contiguous subarrays.

        The cost of a subarray is its first element.
        """
        
        # The first subarray must start with nums[0], so its cost is fixed.
        cost_of_first_subarray = nums[0]
        
        # For the other two subarrays, we need to choose their starting elements
        # from the rest of the array, `nums[1:]`. To minimize the total cost,
        # we must choose the two smallest elements from this remaining part
        # to be the costs of the second and third subarrays.
        
        remaining_elements = nums[1:]
        
        # Sorting the remaining elements makes it easy to find the two smallest.
        # Since n >= 3, remaining_elements will have at least 2 items.
        remaining_elements.sort()
        
        # The minimum costs for the second and third subarrays are these two smallest elements.
        cost_of_second_subarray = remaining_elements[0]
        cost_of_third_subarray = remaining_elements[1]
        
        return cost_of_first_subarray + cost_of_second_subarray + cost_of_third_subarray