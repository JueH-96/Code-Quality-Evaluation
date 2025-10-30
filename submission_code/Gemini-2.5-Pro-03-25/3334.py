import math # This line is not strictly needed for this solution, but might be included in a standard template. Can be removed.
from typing import List # Need this for type hinting

class Solution:
    """
    This class provides a solution to find the minimum number of boxes
    needed to store a given amount of apples, using boxes with the largest
    capacities first.
    """
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        """
        Calculates the minimum number of boxes required to store all apples
        by greedily selecting boxes with the largest capacities.

        The problem requires finding the minimum number of boxes to accommodate
        a total number of apples. Since apples from the same pack can be split,
        we only need to consider the total number of apples and the capacities
        of the available boxes. To minimize the number of boxes, it's optimal
        to use the boxes with the largest capacities first.

        Args:
            apple: A list of integers where apple[i] is the number of apples
                   in the i-th pack. The length of this list is n.
            capacity: A list of integers where capacity[i] is the capacity
                      of the i-th box. The length of this list is m.

        Returns:
            An integer representing the minimum number of boxes needed to
            store all the apples.
        """

        # 1. Calculate the total number of apples that need to be stored.
        # Since apples from the same pack can be distributed into different boxes,
        # we only care about the sum of apples across all packs.
        total_apples = sum(apple)

        # According to the constraints (1 <= apple[i], 1 <= n = len(apple)),
        # total_apples will always be at least 1. Therefore, we don't need
        # a special check for total_apples == 0.

        # 2. Sort the box capacities in descending order.
        # The greedy strategy is to prioritize using boxes with larger capacities
        # first. This ensures that we reach the required total capacity using
        # the minimum possible number of boxes.
        # The sort modifies the list in-place.
        capacity.sort(reverse=True)

        # 3. Iterate through the sorted capacities (from largest to smallest).
        # We will keep track of the accumulated capacity from the selected boxes
        # and the count of boxes selected so far.
        current_accumulated_capacity = 0
        boxes_count = 0
        
        # Iterate through each box capacity, starting from the largest.
        for box_capacity in capacity:
            # Select the current box (which is the largest among the remaining ones)
            # and add its capacity to our running total.
            current_accumulated_capacity += box_capacity
            
            # Increment the count of boxes we have decided to use.
            boxes_count += 1

            # 4. Check if the accumulated capacity is now sufficient.
            # If the total capacity of the boxes selected so far is greater than
            # or equal to the total number of apples required, we have found
            # the minimum number of boxes needed.
            if current_accumulated_capacity >= total_apples:
                # Because we sorted the capacities and picked the largest ones first,
                # this is the minimum number of boxes required. Stop the iteration.
                break

        # 5. Return the final count of boxes used.
        # The problem statement guarantees that it's always possible to redistribute
        # the apples, meaning sum(capacity) >= sum(apple). This ensures that
        # the loop will always find enough capacity and break, returning a valid count.
        return boxes_count