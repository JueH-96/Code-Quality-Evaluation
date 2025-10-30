from typing import List

class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        current_position = 0
        boundary_returns_count = 0
        
        for move_value in nums:
            # The ant's position changes by the value of the current element.
            # If move_value is positive (nums[i] > 0), it moves right by nums[i] units.
            #   This means position increases by nums[i].
            # If move_value is negative (nums[i] < 0), it moves left by -nums[i] units.
            #   This means position decreases by -nums[i], which is equivalent to adding nums[i].
            # So, in both cases, current_position = current_position + move_value.
            current_position += move_value
            
            # We check whether the ant is on the boundary (position 0) 
            # only AFTER it has moved.
            if current_position == 0:
                boundary_returns_count += 1
                
        return boundary_returns_count