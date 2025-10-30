from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        """
        Dynamic programming over the parity (even/odd) of the last picked element.
        
        dp_even  : maximum score of a valid subsequence that ends with an even value
        dp_odd   : same but ends with an odd value
        
        When we consider nums[i] (parity = p):
            - If we append it to a subsequence that already ends with the same
              parity, we pay no penalty and obtain  dp_p + nums[i].
            - If we append it to a subsequence that ends with the opposite
              parity, we pay the penalty x and obtain  dp_{1-p} + nums[i] - x.
            - We can also decide not to take nums[i] at all.
            
        We always keep the best value for each parity.  The answer is the best
        score among the two parities after processing the whole array.
        """
        
        NEG = -10**18               # sufficiently small negative number
        dp_even, dp_odd = NEG, NEG  # best scores ending with even / odd
        
        # The subsequence must start with index 0
        if nums[0] & 1:             # odd
            dp_odd = nums[0]
        else:                       # even
            dp_even = nums[0]
        
        for num in nums[1:]:
            p = num & 1             # 0 -> even, 1 -> odd
            
            # Previous bests (before updating this iteration)
            same_parity_best  = dp_odd if p else dp_even
            diff_parity_best  = dp_even if p else dp_odd
            diff_parity_best -= x    # pay penalty if we switch parity
            
            best_prev = max(same_parity_best, diff_parity_best)
            candidate = best_prev + num
            
            # Update the DP for the current parity
            if p:   # updating dp_odd
                dp_odd = max(dp_odd, candidate)
            else:   # updating dp_even
                dp_even = max(dp_even, candidate)
        
        return max(dp_even, dp_odd)