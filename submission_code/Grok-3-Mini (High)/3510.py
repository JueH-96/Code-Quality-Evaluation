from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        # Sort the maximum heights in descending order
        sorted_max = sorted(maximumHeight, reverse=True)
        
        # Initialize the total sum and the previous height
        total_sum = 0
        prev_height = sorted_max[0]
        total_sum += prev_height
        
        # Iterate through the remaining heights starting from index 1
        for max_val in sorted_max[1:]:
            # Calculate the candidate height
            candidate_height = min(max_val, prev_height - 1)
            # If the candidate height is less than 1, it's impossible
            if candidate_height < 1:
                return -1
            # Add the candidate height to the total sum and update prev_height
            total_sum += candidate_height
            prev_height = candidate_height
        
        # Return the total sum
        return total_sum