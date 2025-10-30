from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        results = []
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            if len(set(subarray)) != k:
                results.append(-1)
                continue
            
            sorted_subarray = sorted(subarray)
            is_consecutive = True
            for j in range(1, k):
                if sorted_subarray[j] != sorted_subarray[j - 1] + 1:
                    is_consecutive = False
                    break
            
            if is_consecutive:
                results.append(sorted_subarray[-1])
            else:
                results.append(-1)
        
        return results