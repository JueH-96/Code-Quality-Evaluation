from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # dp0 = best score ending at an even-valued position
        # dp1 = best score ending at an odd-valued position
        NEG_INF = -10**30
        
        # Initialize dp based on the parity of nums[0]
        if nums[0] % 2 == 0:
            dp0, dp1 = nums[0], NEG_INF
        else:
            dp0, dp1 = NEG_INF, nums[0]
        
        # Process each subsequent position
        for num in nums[1:]:
            if num % 2 == 0:
                # To end at this even num, either come from an even (no penalty)
                # or from an odd (penalty x)
                best_from_even = dp0 + num
                best_from_odd  = dp1 + num - x
                new_dp0 = max(best_from_even, best_from_odd)
                # dp1 stays the same
                dp0 = new_dp0
            else:
                # To end at this odd num, either come from an odd (no penalty)
                # or from an even (penalty x)
                best_from_odd  = dp1 + num
                best_from_even = dp0 + num - x
                new_dp1 = max(best_from_odd, best_from_even)
                # dp0 stays the same
                dp1 = new_dp1
        
        # The answer is the best achievable ending at either parity
        return max(dp0, dp1)