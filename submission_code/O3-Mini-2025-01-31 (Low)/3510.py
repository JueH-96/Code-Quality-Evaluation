from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in descending order.
        sorted_heights = sorted(maximumHeight, reverse=True)
        
        # This variable will keep track of the maximum allowed height for the
        # next assigned tower (needs to be strictly less than the previous assigned height).
        current_max = float('inf')
        
        total_sum = 0
        for height in sorted_heights:
            # For the first element, we can assign its maximum value.
            # For subsequent ones, we must assign a height that is at most one less than the previous assigned height.
            if current_max == float('inf'):
                assigned = height
            else:
                assigned = min(height, current_max - 1)
            
            # If we can't assign a positive integer, return -1.
            if assigned < 1:
                return -1
            
            total_sum += assigned
            # Update current_max for the next tower.
            current_max = assigned
        
        return total_sum

# Example usage:
# sol = Solution()
# print(sol.maximumTotalSum([2,3,4,3]))   # Output: 10
# print(sol.maximumTotalSum([15,10]))     # Output: 25
# print(sol.maximumTotalSum([2,2,1]))      # Output: -1