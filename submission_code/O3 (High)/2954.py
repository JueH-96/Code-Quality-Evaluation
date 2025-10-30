from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if k > n:
            return 0
        
        freq = defaultdict(int)          # frequency of elements in current window
        window_sum = 0                   # sum of current window
        best = 0                         # best (maximum) sum found that satisfies the condition
        
        # Build the first window of size k
        for i in range(k):
            val = nums[i]
            window_sum += val
            freq[val] += 1
        
        # Check the first window
        if len(freq) >= m:
            best = window_sum
        
        # Slide the window across the array
        for right in range(k, n):
            left = right - k              # index that is leaving the window
            
            # Remove the element going out of the window
            left_val = nums[left]
            freq[left_val] -= 1
            if freq[left_val] == 0:
                del freq[left_val]
            window_sum -= left_val
            
            # Add the new element entering the window
            right_val = nums[right]
            window_sum += right_val
            freq[right_val] += 1
            
            # Update best if current window satisfies the distinct count
            if len(freq) >= m and window_sum > best:
                best = window_sum
        
        return best