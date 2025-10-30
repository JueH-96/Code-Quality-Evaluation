from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        """
        Calculates the minimum number of boxes required to store all apples.
        """
        # Step 1: Calculate the total number of apples to be stored.
        # Since apples from a pack can be distributed across boxes, we only need the total sum.
        total_apples = sum(apple)
        
        # Step 2: To minimize the number of boxes, use a greedy approach.
        # Select boxes with the largest capacity first. To do this, sort capacities in descending order.
        capacity.sort(reverse=True)
        
        # Step 3: Iterate through the sorted capacities, adding them up and counting boxes
        # until the accumulated capacity is enough for all apples.
        current_capacity = 0
        for i, cap in enumerate(capacity):
            current_capacity += cap
            # If we have enough capacity, the number of boxes used is the current index + 1
            # (since enumerate provides a 0-based index).
            if current_capacity >= total_apples:
                return i + 1
        
        # According to the problem constraints, a solution is always possible,
        # so the loop is guaranteed to find an answer and return from within the loop.
        # A return statement here would be unreachable.