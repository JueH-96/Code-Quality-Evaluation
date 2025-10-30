from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        max_len = 0
        for i in range(1 << n):
            subsequence = []
            for j in range(n):
                if (i >> j) & 1:
                    subsequence.append(nums[j])
            
            if not subsequence:
                continue
            
            diff_count = 0
            for j in range(len(subsequence) - 1):
                if subsequence[j] != subsequence[j+1]:
                    diff_count += 1
            
            if diff_count <= k:
                max_len = max(max_len, len(subsequence))
        
        return max_len