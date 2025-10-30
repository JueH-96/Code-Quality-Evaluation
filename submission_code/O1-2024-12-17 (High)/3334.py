from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        
        current_sum = 0
        boxes_used = 0
        
        for cap in capacity:
            current_sum += cap
            boxes_used += 1
            if current_sum >= total_apples:
                return boxes_used
        
        return boxes_used