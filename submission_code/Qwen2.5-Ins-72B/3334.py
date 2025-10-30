from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        used_boxes = 0
        for cap in capacity:
            total_apples -= cap
            used_boxes += 1
            if total_apples <= 0:
                return used_boxes