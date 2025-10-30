from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            sub = nums[i:i + k]
            is_increasing = True
            for j in range(k - 1):
                if sub[j] >= sub[j + 1]:
                    is_increasing = False
                    break
            if not is_increasing:
                results.append(-1)
                continue
            diff = sub[-1] - sub[0]
            if diff == k - 1:
                results.append(sub[-1])
            else:
                results.append(-1)
        return results