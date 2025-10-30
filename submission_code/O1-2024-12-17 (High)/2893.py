from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # We track two running values for the best achievable score if
        # the last chosen index had an even-valued number (bestEven)
        # or had an odd-valued number (bestOdd).
        bestEven = float('-inf')
        bestOdd = float('-inf')
        
        # Initialize based on the parity of nums[0]
        if nums[0] % 2 == 0:
            bestEven = nums[0]
        else:
            bestOdd = nums[0]
        
        # Iterate through the array, updating the best possible scores
        # depending on the parity of the current number.
        for i in range(1, len(nums)):
            if nums[i] % 2 == 0:
                # If current value is even, either continue from bestEven (same parity, no penalty)
                # or come from bestOdd (different parity, penalty x).
                dpVal = max(bestEven + nums[i], bestOdd + nums[i] - x)
                bestEven = max(bestEven, dpVal)
            else:
                # If current value is odd, either continue from bestOdd (same parity, no penalty)
                # or come from bestEven (different parity, penalty x).
                dpVal = max(bestOdd + nums[i], bestEven + nums[i] - x)
                bestOdd = max(bestOdd, dpVal)
        
        # The answer is the best score we can achieve ending at any parity.
        return max(bestEven, bestOdd)