from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if k > n:
            return 0
        
        max_sum = 0
        current_sum = 0
        freq_map = defaultdict(int)
        
        # Initialize the first window
        for i in range(k):
            current_sum += nums[i]
            freq_map[nums[i]] += 1
        
        distinct = len(freq_map)
        if distinct >= m:
            max_sum = current_sum
        
        # Slide the window through the rest of the array
        for i in range(1, n - k + 1):
            # Update current_sum by removing the outgoing element and adding the incoming element
            outgoing = nums[i - 1]
            incoming = nums[i + k - 1]
            current_sum = current_sum - outgoing + incoming
            
            # Update frequency map for outgoing element
            freq_map[outgoing] -= 1
            if freq_map[outgoing] == 0:
                del freq_map[outgoing]
            
            # Update frequency map for incoming element
            freq_map[incoming] += 1
            
            # Check the number of distinct elements in the current window
            distinct = len(freq_map)
            if distinct >= m and current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum