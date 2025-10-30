from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        # Use a frequency dictionary to count elements in the current window.
        freq = {}
        current_sum = 0
        max_sum = 0
        
        # Initialize the first window of size k.
        for i in range(k):
            current_sum += nums[i]
            freq[nums[i]] = freq.get(nums[i], 0) + 1
        
        # Check if this window has at least m distinct elements.
        if len(freq) >= m:
            max_sum = current_sum
        
        # Slide the window through the array.
        for i in range(k, n):
            # Remove the element that is leaving the window.
            left_elem = nums[i - k]
            freq[left_elem] -= 1
            if freq[left_elem] == 0:
                del freq[left_elem]
            current_sum -= left_elem
            
            # Add the new element entering the window.
            current_sum += nums[i]
            freq[nums[i]] = freq.get(nums[i], 0) + 1
            
            # Update the max_sum if the current window satisfies the condition.
            if len(freq) >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum