from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in descending order
        sorted_heights = sorted(enumerate(maximumHeight), key=lambda x: x[1], reverse=True)
        
        # Initialize the total sum and the set of used heights
        total_sum = 0
        used_heights = set()
        
        # Iterate over the sorted heights
        for i, height in sorted_heights:
            # Find the maximum available height for the current tower
            for h in range(height, 0, -1):
                if h not in used_heights:
                    # Assign the available height to the current tower
                    total_sum += h
                    used_heights.add(h)
                    break
            else:
                # If no available height is found, return -1
                return -1
        
        return total_sum