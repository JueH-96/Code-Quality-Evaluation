from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Iterate over all possible subarrays
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                
                # Check if the subarray is alternating
                is_alternating = True
                for k in range(len(subarray) - 1):
                    if subarray[k] == subarray[k+1]:
                        is_alternating = False
                        break
                
                # If the subarray is alternating, increment the count
                if is_alternating:
                    count += 1
        
        return count