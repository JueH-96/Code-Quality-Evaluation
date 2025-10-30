from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_length = -1
        
        # iterate over possible starting indices
        for i in range(n - 1):
            # check the first condition for a possible alternating subarray
            if nums[i+1] != nums[i] + 1:
                continue
            
            # We have a valid starting segment of length 2
            current_length = 2
            # s0 and s1 are determined
            s0, s1 = nums[i], nums[i] + 1
            
            # continue trying to extend the alternating subarray
            while i + current_length < n:
                # For even index in the subarray (0-indexed relative to i), expect s0, for odd expect s1.
                if current_length % 2 == 0:
                    expected = s0
                else:
                    expected = s1
                
                if nums[i + current_length] == expected:
                    current_length += 1
                else:
                    break
                    
            # update maximum found length
            max_length = max(max_length, current_length)
        
        return max_length