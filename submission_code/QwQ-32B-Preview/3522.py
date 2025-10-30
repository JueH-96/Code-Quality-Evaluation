from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            all_consecutive = True
            for j in range(k - 1):
                if nums[i + j + 1] - nums[i + j] != 1:
                    all_consecutive = False
                    break
            if all_consecutive:
                results.append(nums[i + k - 1])
            else:
                results.append(-1)
        return results