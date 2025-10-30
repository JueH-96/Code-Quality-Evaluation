from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        # Sort the capacities in descending order to use the largest boxes first
        capacity.sort(reverse=True)

        # Calculate the total number of apples
        total_apples = sum(apple)

        # Initialize the number of boxes needed
        boxes_needed = 0

        # Initialize the current capacity used
        current_capacity = 0

        # Iterate through the capacities
        for cap in capacity:
            # Add the capacity of the current box to the current capacity
            current_capacity += cap
            # Increment the number of boxes needed
            boxes_needed += 1
            # If the current capacity is greater than or equal to the total number of apples, break
            if current_capacity >= total_apples:
                break

        return boxes_needed