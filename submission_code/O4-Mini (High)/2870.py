from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        ans = -1
        
        # Try every starting position i
        for i in range(n - 1):
            # Check the first required difference +1
            if nums[i+1] == nums[i] + 1:
                cur_len = 2
                last_diff = 1  # the last difference we saw (nums[i+1]-nums[i])
                
                # Extend the alternating pattern as far as possible
                for j in range(i + 2, n):
                    diff = nums[j] - nums[j-1]
                    # We need diff to equal -last_diff to alternate
                    if diff == -last_diff:
                        cur_len += 1
                        last_diff = diff
                    else:
                        break
                
                ans = max(ans, cur_len)
        
        return ans