from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        # suffixDistinct[i] = number of distinct elements in nums[i..n-1]
        suffixDistinct = [0] * (n + 1)
        seen = set()
        for i in range(n - 1, -1, -1):
            seen.add(nums[i])
            suffixDistinct[i] = len(seen)
        
        res = []
        prefixSeen = set()
        prefixCount = 0
        for i, x in enumerate(nums):
            if x not in prefixSeen:
                prefixSeen.add(x)
                prefixCount += 1
            # suffix after index i is nums[i+1..n-1], whose distinct count is suffixDistinct[i+1]
            suffixCount = suffixDistinct[i + 1]
            res.append(prefixCount - suffixCount)
        
        return res