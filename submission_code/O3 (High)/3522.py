from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        res = []
        
        # Iterate over every window of size k
        for start in range(n - k + 1):
            valid = True
            
            # Check if the current window is strictly increasing
            # by exactly 1 at every step
            for j in range(start, start + k - 1):
                if nums[j + 1] != nums[j] + 1:
                    valid = False
                    break
            
            # Append the power of this sub-array
            if valid:
                res.append(nums[start + k - 1])  # maximum element in an ascending window
            else:
                res.append(-1)
        
        return res