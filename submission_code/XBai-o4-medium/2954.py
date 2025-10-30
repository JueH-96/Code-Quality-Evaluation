from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if k > n:
            return 0
        
        current_sum = sum(nums[:k])
        max_sum = 0
        
        freq = {}
        for num in nums[:k]:
            freq[num] = freq.get(num, 0) + 1
        
        if len(freq) >= m:
            max_sum = current_sum
        
        for i in range(1, n - k + 1):
            left = nums[i - 1]
            right = nums[i + k - 1]
            
            # Update current sum
            current_sum = current_sum - left + right
            
            # Update frequency for the left element
            freq[left] -= 1
            if freq[left] == 0:
                del freq[left]
            
            # Update frequency for the right element
            freq[right] = freq.get(right, 0) + 1
            
            # Check current distinct count
            current_distinct = len(freq)
            if current_distinct >= m and current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum