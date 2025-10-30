from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximumHeights in descending order to assign the largest possible values first
        sortedHeights = sorted(maximumHeight, reverse=True)
        
        total = 0
        prev = float('inf')  # Initialize previous height to a very large number
        
        for h in sortedHeights:
            # Assign the current height as the minimum of current max and previous height minus 1
            current = min(h, prev - 1)
            if current <= 0:
                return -1  # If current height is not positive, return -1
            total += current
            prev = current  # Update previous height to current for next iteration
        
        return total  # Return the accumulated total sum