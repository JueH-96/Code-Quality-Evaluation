import collections
from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        # Sort the array. This is crucial for the sliding window approach to work,
        # as it allows us to easily check the 'k' constraint with the leftmost element.
        nums.sort()

        n = len(nums)
        left = 0  # Left pointer of the sliding window
        max_freq = 0  # Stores the maximum frequency found so far

        # Iterate the 'right' pointer through the array
        for right in range(n):
            # The current element nums[right] is chosen as the target value for the window
            # nums[left...right]. We want to make all elements in this window equal to nums[right].

            # We need to satisfy two conditions for this window to be valid:
            # 1. Value Range Constraint: All elements nums[i] (left <= i <= right) must be
            #    transformable to nums[right] within the range [-k, k]. Since nums is sorted,
            #    this is equivalent to checking if nums[right] - nums[left] <= k.
            #    If this holds, then for any nums[i] in the window, nums[right] - nums[i] <= nums[right] - nums[left] <= k.
            # 2. Operations Count Constraint: The number of elements that are not nums[right]
            #    within the window (i.e., nums[left] through nums[right-1]) and thus might need
            #    modification must not exceed `numOperations`. This count is `right - left`.

            # Shrink the window from the left if either condition is violated.
            # This 'while' loop ensures that the window always remains valid.
            while (nums[right] - nums[left] > k) or \
                  (right - left > numOperations):
                left += 1
            
            # After the while loop, the window nums[left...right] is valid.
            # Its size is (right - left + 1). This is a candidate for the maximum frequency.
            max_freq = max(max_freq, right - left + 1)
            
        return max_freq