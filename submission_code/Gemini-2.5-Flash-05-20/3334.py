from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        # 1. Calculate the total number of apples from all packs.
        total_apples = sum(apple)
        
        # 2. Sort the capacities of the boxes in descending order.
        # This greedy strategy ensures we pick the largest boxes first,
        # which will minimize the count of boxes needed to reach the total capacity.
        capacity.sort(reverse=True)
        
        # 3. Initialize variables to keep track of the accumulated capacity
        # and the number of boxes selected.
        current_capacity_sum = 0
        boxes_needed = 0
        
        # 4. Iterate through the sorted capacities.
        for box_cap in capacity:
            # Add the current box's capacity to the sum.
            current_capacity_sum += box_cap
            # Increment the count of boxes used.
            boxes_needed += 1
            
            # Check if the accumulated capacity is sufficient to hold all apples.
            if current_capacity_sum >= total_apples:
                # If yes, we have found the minimum number of boxes, so we can stop.
                break
                
        # 5. Return the total number of boxes needed.
        return boxes_needed