from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in descending order
        maximumHeight.sort(reverse=True)
        
        total = 0
        # prev_height is the maximum height we can assign to the next tower
        prev_height = 10**18  # sufficiently large sentinel
        
        for mh in maximumHeight:
            # We pick the largest valid height <= mh and < prev_height
            curr = min(mh, prev_height - 1)
            # If we cannot pick a positive height, it's impossible
            if curr <= 0:
                return -1
            total += curr
            prev_height = curr
        
        return total