from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Try every non-empty subarray [l..r]
        for l in range(n):
            for r in range(l, n):
                # Form the array after removing nums[l..r]
                remaining = nums[:l] + nums[r+1:]
                
                # Check if 'remaining' is strictly increasing
                ok = True
                for i in range(1, len(remaining)):
                    if remaining[i] <= remaining[i-1]:
                        ok = False
                        break
                
                if ok:
                    count += 1
        
        return count