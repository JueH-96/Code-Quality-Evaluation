from typing import List
from collections import defaultdict

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total = n * (n + 1) // 2
        k = (total - 1) // 2  # Zero-based index of the median
        
        left, right = 1, n
        
        while left < right:
            mid = (left + right) // 2
            count = self.count_subarrays_with_at_most(mid, nums)
            if count > k:
                right = mid
            else:
                left = mid + 1
        return left
    
    def count_subarrays_with_at_most(self, m: int, nums: List[int]) -> int:
        count = 0
        left = 0
        freq = defaultdict(int)
        distinct = 0
        for right in range(len(nums)):
            num = nums[right]
            if freq[num] == 0:
                distinct += 1
            freq[num] += 1
            # Shrink the window until distinct <= m
            while distinct > m:
                left_num = nums[left]
                freq[left_num] -= 1
                if freq[left_num] == 0:
                    distinct -= 1
                left += 1
            # Add the number of valid subarrays ending at right
            count += right - left + 1
        return count