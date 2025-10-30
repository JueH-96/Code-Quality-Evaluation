from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # Initialize best scores based on parity of first element.
        if nums[0] % 2 == 0:
            bestEven = nums[0]
            bestOdd = -math.inf
        else:
            bestOdd = nums[0]
            bestEven = -math.inf

        # Iterate over the rest of elements.
        for num in nums[1:]:
            if num % 2 == 0:
                # For even number, we can come from an even with no cost, or from an odd with penalty.
                newBestEven = max(bestEven + num, bestOdd - x + num)
                # Even number won't update bestOdd because we are not adding odd.
                bestEven = newBestEven
            else:
                # For odd number, we can come from odd with no cost or from even with penalty.
                newBestOdd = max(bestOdd + num, bestEven - x + num)
                bestOdd = newBestOdd

        # The answer is the maximum score available from either parity.
        return max(bestEven, bestOdd)