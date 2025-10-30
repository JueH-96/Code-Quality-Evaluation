from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity_sorted = sorted(capacity, reverse=True)
        boxes_used = 0
        current_capacity = 0
        for cap in capacity_sorted:
            current_capacity += cap
            boxes_used += 1
            if current_capacity >= total_apples:
                break
        return boxes_used