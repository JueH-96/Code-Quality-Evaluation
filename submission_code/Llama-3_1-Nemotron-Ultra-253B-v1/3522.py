from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n - k + 1):
            sub = nums[i:i+k]
            is_sorted = True
            for j in range(len(sub)-1):
                if sub[j] >= sub[j+1]:
                    is_sorted = False
                    break
            if not is_sorted:
                result.append(-1)
            else:
                if sub[-1] - sub[0] == k - 1:
                    result.append(sub[-1])
                else:
                    result.append(-1)
        return result