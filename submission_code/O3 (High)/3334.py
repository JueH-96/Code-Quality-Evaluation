from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        """
        To minimize the number of boxes we only need to make sure that
        the total capacity of the chosen boxes is at least the total
        number of apples.  
        The greedy strategy of always picking the largest‐capacity boxes
        first achieves this with the fewest boxes.
        """
        total_apples = sum(apple)
        # Sort capacities from largest to smallest
        capacity.sort(reverse=True)
        
        current = 0          # accumulated capacity
        boxes_used = 0       # number of boxes taken so far
        
        for cap in capacity:
            current += cap
            boxes_used += 1
            if current >= total_apples:
                return boxes_used
        
        # According to the problem statement this line is never reached
        # because it is guaranteed that redistribution is possible.
        return boxes_used