import collections
from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        """
        Calculates the maximum sum of an "almost unique" subarray of length k.

        This solution uses a sliding window of size k to efficiently process
        the subarrays. A hash map tracks the frequency of elements within the
        current window, allowing for quick updates to the window's sum and
        its number of distinct elements.

        Time Complexity: O(N), where N is the length of nums.
        Space Complexity: O(k), for the hash map storing element counts.
        """
        
        n = len(nums)
        
        max_sum = 0
        window_sum = 0
        counts = collections.defaultdict(int)
        
        # 1. Initialize the first window of size k
        # We iterate from 0 to k-1 to build the initial state.
        for i in range(k):
            num = nums[i]
            window_sum += num
            counts[num] += 1
        
        # Check if the first window meets the criteria
        if len(counts) >= m:
            max_sum = window_sum
            
        # 2. Slide the window from index k to the end of the array
        # In each step, we add a new element and remove the leftmost element.
        for i in range(k, n):
            # Add the new element entering the window from the right
            new_num = nums[i]
            window_sum += new_num
            counts[new_num] += 1
            
            # Remove the old element leaving the window from the left
            old_num = nums[i-k]
            window_sum -= old_num
            counts[old_num] -= 1
            
            # If the count of the old element becomes zero, it's no longer
            # in the window, so we remove it from our counts map to keep
            # the distinct count (len(counts)) accurate.
            if counts[old_num] == 0:
                del counts[old_num]
            
            # If the current window meets the criteria, update the maximum sum
            if len(counts) >= m:
                max_sum = max(max_sum, window_sum)
                
        return max_sum