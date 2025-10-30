from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        count = 0
        n = len(nums)
        
        for i in range(n):
            and_val = nums[i]
            if and_val == k:
                count += 1
            for j in range(i + 1, n):
                and_val &= nums[j]
                if and_val == k:
                    count += 1
                elif and_val < k:
                    break
        return count