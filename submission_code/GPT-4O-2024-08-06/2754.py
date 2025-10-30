from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        # Separate positive, negative, and zero numbers
        positives = [x for x in nums if x > 0]
        negatives = [x for x in nums if x < 0]
        zeros = [x for x in nums if x == 0]
        
        # Sort negatives to easily drop the smallest one if needed
        negatives.sort()
        
        # If the number of negatives is odd, drop the smallest (least negative) one
        if len(negatives) % 2 != 0:
            negatives.pop()
        
        # Calculate the product of all remaining numbers
        product = 1
        for num in positives + negatives:
            product *= num
        
        # If there are no positives and no negatives, it means all numbers are zero
        # In this case, the maximum strength is 0
        if not positives and not negatives:
            return 0
        
        return product