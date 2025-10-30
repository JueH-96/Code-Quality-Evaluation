from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        total_distinct = len(set(nums))
        freq = {}
        distinct = 0
        total_subarrays = 0
        right = 0
        
        for left in range(n):
            while distinct < total_distinct and right < n:
                num = nums[right]
                freq[num] = freq.get(num, 0) + 1
                if freq[num] == 1:
                    distinct += 1
                right += 1
            
            if distinct == total_distinct:
                total_subarrays += (n - right + 1)
            
            num_left = nums[left]
            freq[num_left] -= 1
            if freq[num_left] == 0:
                distinct -= 1
        
        return total_subarrays