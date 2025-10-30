from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        res = []
        for i in range(n - k + 1):
            valid = True
            for j in range(i, i + k - 1):
                if nums[j + 1] != nums[j] + 1:
                    valid = False
                    break
            res.append(nums[i + k - 1] if valid else -1)
        return res