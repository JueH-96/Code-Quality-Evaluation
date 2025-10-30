from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        # Sort the capacity array in descending order
        capacity.sort(reverse=True)
        
        # Initialize the total apples and the number of boxes used
        total_apples = sum(apple)
        boxes_used = 0
        
        # Initialize the index for the capacity array
        capacity_index = 0
        
        # Iterate over the capacity array
        while total_apples > 0:
            # If the current capacity is greater than or equal to the total apples, 
            # we can fill the current box and break the loop
            if capacity[capacity_index] >= total_apples:
                boxes_used += 1
                break
            # Otherwise, fill the current box and move to the next box
            else:
                total_apples -= capacity[capacity_index]
                boxes_used += 1
                capacity_index += 1
        
        return boxes_used