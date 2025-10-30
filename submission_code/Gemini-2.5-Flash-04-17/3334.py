from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        """
        Calculates the minimum number of boxes required to hold all apples.

        Args:
            apple: A list of integers representing the number of apples in each pack.
            capacity: A list of integers representing the capacity of each box.

        Returns:
            The minimum number of boxes needed.
        """
        # Calculate the total number of apples to be stored.
        total_apples = sum(apple)

        # Sort box capacities in descending order. This ensures we use the boxes
        # with the largest capacities first, which is the greedy approach
        # that minimizes the number of boxes used.
        sorted_capacity = sorted(capacity, reverse=True)

        # Iterate through the sorted capacities, accumulating the capacity
        # and counting the boxes until the total capacity is sufficient
        # to hold all the apples.
        current_capacity_sum = 0
        boxes_count = 0
        for cap in sorted_capacity:
            current_capacity_sum += cap
            boxes_count += 1
            # If the current accumulated capacity is greater than or equal to the
            # total number of apples, we have found the minimum number of boxes.
            if current_capacity_sum >= total_apples:
                break

        return boxes_count