import collections
from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # Sort the array in ascending order.
        # This allows us to easily access the largest elements and ensures
        # that when we consider nums[i] as the longest side, all preceding
        # elements nums[0]...nums[i-1] are smaller or equal.
        nums.sort()
        
        n = len(nums)
        
        # Calculate the sum of all elements.
        # This sum will be progressively reduced as we remove elements
        # that do not satisfy the polygon condition.
        current_sum = sum(nums)
        
        # Iterate from the largest element (potential longest side) downwards.
        # We need at least 3 sides for a polygon.
        # The loop variable 'i' represents the index of the current largest
        # candidate side (nums[i]).
        # The loop runs from n-1 down to 2 (inclusive).
        # When i=2, we are considering nums[0], nums[1], nums[2].
        for i in range(n - 1, 1, -1):
            # nums[i] is the current candidate for the longest side.
            # current_sum, at this point, holds the sum of nums[0] through nums[i].
            # (In the first iteration, it's sum of all nums; in subsequent iterations,
            # it's the sum of elements not yet discarded).
            
            # The sum of the other sides (nums[0]...nums[i-1]) is current_sum - nums[i].
            # Check the polygon condition: sum of other sides > longest side.
            if current_sum - nums[i] > nums[i]:
                # If the condition is met, we have found a valid polygon.
                # Since we are iterating from the largest possible set of sides
                # (starting with all elements, then n-1 elements, etc.),
                # the first time this condition is true, current_sum represents
                # the largest possible perimeter.
                return current_sum
            else:
                # If the condition is not met, nums[i] is too large to be the longest side
                # of a polygon formed by the current set of elements (nums[0]...nums[i]).
                # We must discard nums[i] and try to form a polygon with the remaining
                # smaller elements (nums[0]...nums[i-1]).
                current_sum -= nums[i]
        
        # If the loop finishes without finding any valid polygon,
        # it means no combination of sides (at least 3) can form a polygon.
        return -1