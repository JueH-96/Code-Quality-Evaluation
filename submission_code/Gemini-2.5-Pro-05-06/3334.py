from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        
        total_apples = sum(apple)
        
        # Constraints:
        # 1 <= n == apple.length <= 50
        # 1 <= apple[i] <= 50
        # This means total_apples will always be >= 1.
        # So, no specific check for total_apples == 0 is needed.
        
        # Sort capacities in descending order.
        # This ensures that we greedily pick boxes with the largest capacity first.
        capacity.sort(reverse=True)
        
        boxes_needed = 0
        current_accumulated_capacity = 0
        
        # Iterate through the sorted capacities and accumulate capacity
        # until it's enough to hold all apples.
        for box_capacity in capacity:
            current_accumulated_capacity += box_capacity
            boxes_needed += 1
            if current_accumulated_capacity >= total_apples:
                # We have found the minimum number of boxes.
                return boxes_needed
        
        # Based on the problem statement:
        # "The input is generated such that it's possible to redistribute packs of apples into boxes."
        # This implies that the sum of all capacities in the 'capacity' array
        # is greater than or equal to 'total_apples'.
        # Therefore, the loop above is guaranteed to find sufficient capacity and return.
        # This part of the code should theoretically be unreachable.
        return boxes_needed # This line is technically unreachable under problem constraints.
                           # If it were reachable, it would mean all boxes were used.