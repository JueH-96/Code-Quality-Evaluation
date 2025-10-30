from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if k > n:
            return 0
        
        max_sum = 0
        current_sum = sum(nums[:k])
        freq = defaultdict(int)
        for num in nums[:k]:
            freq[num] += 1
        distinct = len(freq)
        
        if distinct >= m:
            max_sum = current_sum
        
        for i in range(1, n - k + 1):
            # Remove the element leaving the window
            left_num = nums[i - 1]
            freq[left_num] -= 1
            if freq[left_num] == 0:
                del freq[left_num]
                distinct -= 1
            
            # Add the new element entering the window
            right_num = nums[i + k - 1]
            if freq[right_num] == 0:
                distinct += 1
            freq[right_num] += 1
            
            # Update current_sum
            current_sum = current_sum - left_num + right_num
            
            # Check if current window is valid
            if distinct >= m and current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum if max_sum != 0 else 0