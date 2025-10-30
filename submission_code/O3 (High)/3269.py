from typing import List

class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n = len(nums)
        m = len(pattern)
        
        # Pre–compute the sign of every consecutive pair in nums
        #  1  -> increasing
        #  0  -> equal
        # -1  -> decreasing
        signs = []
        for i in range(n - 1):
            if nums[i + 1] > nums[i]:
                signs.append(1)
            elif nums[i + 1] == nums[i]:
                signs.append(0)
            else:
                signs.append(-1)
        
        # Count how many windows of length m in `signs` match `pattern`
        ans = 0
        for start in range(n - m):          # there are n - (m + 1) + 1 == n - m such windows
            if signs[start:start + m] == pattern:
                ans += 1
        return ans