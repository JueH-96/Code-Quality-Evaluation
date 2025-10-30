from typing import List

class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        m = len(pattern)
        required_length = m + 1
        count = 0
        max_i = n - required_length
        
        for i in range(max_i + 1):
            valid = True
            for k in range(m):
                a = nums[i + k]
                b = nums[i + k + 1]
                if pattern[k] == 1:
                    if b <= a:
                        valid = False
                        break
                elif pattern[k] == 0:
                    if a != b:
                        valid = False
                        break
                elif pattern[k] == -1:
                    if b >= a:
                        valid = False
                        break
            if valid:
                count += 1
        return count