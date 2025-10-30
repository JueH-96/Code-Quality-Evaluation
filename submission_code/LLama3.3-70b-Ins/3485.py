from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        # Sort the start array
        start.sort()
        
        # Initialize the minimum and maximum possible scores
        low, high = 0, d
        
        # Perform binary search to find the maximum possible score
        while low < high:
            mid = (low + high + 1) // 2
            if self.is_valid(start, d, mid):
                low = mid
            else:
                high = mid - 1
        
        return low
    
    def is_valid(self, start: List[int], d: int, score: int) -> bool:
        # Initialize the previous chosen integer
        prev = float('-inf')
        
        # Iterate over the start array
        for s in start:
            # If the current interval does not overlap with the previous chosen integer, 
            # update the previous chosen integer to be the start of the current interval plus the score
            if prev + score <= s:
                prev = s + score
            # If the current interval overlaps with the previous chosen integer, 
            # update the previous chosen integer to be the end of the current interval
            elif prev + score > s + d:
                return False
            # If the current interval overlaps with the previous chosen integer, 
            # update the previous chosen integer to be the maximum of the current previous chosen integer 
            # and the start of the current interval
            else:
                prev = max(prev, s)
        
        return True