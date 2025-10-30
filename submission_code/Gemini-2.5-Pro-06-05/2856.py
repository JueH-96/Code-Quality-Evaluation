from typing import List
from collections import Counter

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        """
        Calculates the number of complete subarrays using an efficient sliding window approach.
        A complete subarray has the same number of distinct elements as the entire array.
        """
        # The number of distinct elements in the whole array.
        # A subarray is "complete" if it has this many distinct elements.
        k = len(set(nums))
        n = len(nums)
        
        res = 0
        left = 0
        # Use a Counter to store frequencies of elements in the current window.
        window = Counter()

        # Use a sliding window approach. `right` expands the window.
        for right in range(n):
            window[nums[right]] += 1
            
            # While the window is complete, any subarray starting at `left`
            # and ending at or after `right` is also complete.
            # We count these subarrays and then shrink the window from the left.
            while len(window) == k:
                # Number of complete subarrays that start at index `left` is `n - right`.
                # This is because any subarray nums[left...j] where j >= right is complete.
                res += n - right
                
                # Shrink the window from the left.
                window[nums[left]] -= 1
                if window[nums[left]] == 0:
                    # Remove the element from the window's keys if its count is zero
                    # to ensure len(window) correctly reflects distinct elements.
                    del window[nums[left]]
                left += 1
                
        return res