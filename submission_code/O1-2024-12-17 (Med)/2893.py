class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # We maintain two running values:
        # bestEven = maximum score achievable so far ending in an even-valued position
        # bestOdd  = maximum score achievable so far ending in an odd-valued position
        
        import math
        bestEven = -math.inf
        bestOdd = -math.inf
        
        # Initialize with the first element
        if nums[0] % 2 == 0:
            bestEven = nums[0]
        else:
            bestOdd = nums[0]
        
        # Traverse the array from index 1 onwards
        for i in range(1, len(nums)):
            # If current nums[i] is even:
            if nums[i] % 2 == 0:
                newEven = max(bestEven + nums[i], bestOdd + nums[i] - x)
                bestEven = newEven
            else:
                # If current nums[i] is odd:
                newOdd = max(bestOdd + nums[i], bestEven + nums[i] - x)
                bestOdd = newOdd
        
        # The answer is the best possible score ending in either an even or odd position
        return max(bestEven, bestOdd)