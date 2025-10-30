from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        best = 0
        
        # Try every starting index i
        for i in range(n - 1):
            # The first step must increase by 1
            if nums[i+1] != nums[i] + 1:
                continue
            
            # We have at least a length-2 alternating subarray
            length = 2
            # Extend j from i+2 onward
            for j in range(i+2, n):
                # Compute expected difference
                # If (j-i) is odd, we expect +1; if even, -1
                if ((j - i) % 2 == 1 and nums[j] - nums[j-1] == 1) or \
                   ((j - i) % 2 == 0 and nums[j] - nums[j-1] == -1):
                    length += 1
                else:
                    break
            
            best = max(best, length)
        
        # If no valid alternating subarray found, return -1
        return best if best > 1 else -1