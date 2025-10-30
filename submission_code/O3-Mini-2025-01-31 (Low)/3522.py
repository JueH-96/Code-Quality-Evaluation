from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            sub = nums[i:i+k]
            valid = True
            for j in range(len(sub)-1):
                if sub[j+1] - sub[j] != 1:
                    valid = False
                    break
            if valid:
                results.append(sub[-1])
            else:
                results.append(-1)
        return results