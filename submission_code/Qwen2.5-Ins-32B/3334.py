from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        boxes_needed = 0
        current_capacity = 0
        
        for box in capacity:
            current_capacity += box
            boxes_needed += 1
            if current_capacity >= total_apples:
                break
        
        return boxes_needed