from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n - k + 1):
            valid = True
            for j in range(i, i + k - 1):
                if nums[j + 1] - nums[j] != 1:
                    valid = False
                    break
            if valid:
                result.append(nums[i + k - 1])
            else:
                result.append(-1)
        return result