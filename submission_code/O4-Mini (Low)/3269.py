from typing import List

class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        """
        Count the number of subarrays of length len(pattern)+1 in nums that match the given pattern.
        pattern[k] == 1  -> nums[i+k+1] > nums[i+k]
        pattern[k] == 0  -> nums[i+k+1] == nums[i+k]
        pattern[k] == -1 -> nums[i+k+1] < nums[i+k]
        """
        n = len(nums)
        m = len(pattern)
        count = 0
        
        # We need windows of size m+1, so starting positions go from 0 to n-(m+1)
        for i in range(n - m):
            match = True
            for k, p in enumerate(pattern):
                current = nums[i + k]
                nxt = nums[i + k + 1]
                
                if p == 1:
                    if not (nxt > current):
                        match = False
                        break
                elif p == 0:
                    if not (nxt == current):
                        match = False
                        break
                else:  # p == -1
                    if not (nxt < current):
                        match = False
                        break
            
            if match:
                count += 1
        
        return count