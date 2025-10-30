from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count_even = 0
        count_odd = 0
        
        for num in nums:
            if num % 2 == 0:
                count_even += 1
            else:
                count_odd += 1
        
        transitions = 0
        n = len(nums)
        if n == 0:
            return 0
        
        prev = nums[0] % 2
        for i in range(1, n):
            curr = nums[i] % 2
            if curr != prev:
                transitions += 1
            prev = curr
        
        max_same = max(count_even, count_odd)
        max_alt = transitions + 1
        
        return max(max_same, max_alt)