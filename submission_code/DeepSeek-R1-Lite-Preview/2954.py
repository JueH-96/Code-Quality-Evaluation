from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        freq = {}
        current_sum = 0
        max_sum = 0
        distinct = 0
        n = len(nums)
        
        # Build the initial window
        for i in range(k):
            current_sum += nums[i]
            freq[nums[i]] = freq.get(nums[i], 0) + 1
            if freq[nums[i]] == 1:
                distinct += 1
        if distinct >= m:
            max_sum = current_sum
        
        # Slide the window
        for i in range(k, n):
            outgoing = nums[i - k]
            current_sum -= outgoing
            freq[outgoing] -= 1
            if freq[outgoing] == 0:
                distinct -= 1
                del freq[outgoing]
            
            incoming = nums[i]
            current_sum += incoming
            freq[incoming] = freq.get(incoming, 0) + 1
            if freq[incoming] == 1:
                distinct += 1
            
            if distinct >= m:
                if current_sum > max_sum:
                    max_sum = current_sum
        
        return max_sum