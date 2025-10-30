from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        def is_consecutive_sorted(subarray):
            return subarray == list(range(subarray[0], subarray[0] + k))
        
        results = []
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i + k]
            if is_consecutive_sorted(sorted(subarray)):
                results.append(max(subarray))
            else:
                results.append(-1)
        
        return results