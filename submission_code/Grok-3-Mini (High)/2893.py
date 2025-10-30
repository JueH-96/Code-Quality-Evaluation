from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        NEG_INF = -10**12 - 1
        # Initialize for i=0
        parity_0 = nums[0] % 2
        dp_val = nums[0]
        ans = dp_val
        if parity_0 == 0:
            max_even = dp_val
            max_odd = NEG_INF
        else:
            max_odd = dp_val
            max_even = NEG_INF
        
        for i in range(1, n):
            parity_i = nums[i] % 2
            if parity_i == 0:  # even
                opt_same = max_even
                opt_diff = max_odd - x if max_odd != NEG_INF else NEG_INF
                dp_val_new = nums[i] + max(opt_same, opt_diff)
            else:  # odd
                opt_same = max_odd
                opt_diff = max_even - x if max_even != NEG_INF else NEG_INF
                dp_val_new = nums[i] + max(opt_same, opt_diff)
            
            # Update ans
            ans = max(ans, dp_val_new)
            
            # Update cumulative max
            if parity_i == 0:
                max_even = max(max_even, dp_val_new)
            else:
                max_odd = max(max_odd, dp_val_new)
        
        return ans