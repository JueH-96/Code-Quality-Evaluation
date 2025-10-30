from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in non-decreasing order
        maximumHeight.sort()
        
        total = 0
        # 'last' is the height assigned to the next tower to the right.
        # We start with a very large number so that the largest tower
        # can take its full maximum.
        last = 10**18
        
        # Iterate from the largest maximumHeight down to the smallest
        for m in reversed(maximumHeight):
            # We must pick a height strictly less than 'last' to keep uniqueness,
            # but no more than m.
            assign = min(m, last - 1)
            # If we cannot assign a positive height, it's impossible.
            if assign <= 0:
                return -1
            total += assign
            last = assign
        
        return total