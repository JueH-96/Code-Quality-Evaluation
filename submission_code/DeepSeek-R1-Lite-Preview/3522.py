from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            if subarray == sorted(subarray):
                sub_max = max(subarray)
                sub_min = min(subarray)
                if sub_max - sub_min + 1 == k:
                    results.append(sub_max)
                else:
                    results.append(-1)
            else:
                results.append(-1)
        
        return results