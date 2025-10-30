from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        results = []
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i+k]
            if subarray == sorted(subarray) and all(subarray[j] + 1 == subarray[j+1] for j in range(len(subarray)-1)):
                results.append(subarray[-1])
            else:
                results.append(-1)
        return results