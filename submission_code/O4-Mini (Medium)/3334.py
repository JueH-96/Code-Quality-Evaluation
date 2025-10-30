from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        # Sort boxes by capacity in descending order
        capacity.sort(reverse=True)
        
        used = 0
        current_capacity = 0
        
        # Pick largest boxes until we can hold all apples
        for cap in capacity:
            current_capacity += cap
            used += 1
            if current_capacity >= total_apples:
                return used
        
        # Problem statement guarantees it's always possible
        return used