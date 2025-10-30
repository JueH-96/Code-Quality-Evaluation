from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        res = []
        
        # Slide a window of size k over nums
        for i in range(n - k + 1):
            valid = True
            # Check consecutive ascending property
            for j in range(i + 1, i + k):
                if nums[j] - nums[j - 1] != 1:
                    valid = False
                    break
            # If valid, the power is the max element (last in window), else -1
            res.append(nums[i + k - 1] if valid else -1)
        
        return res