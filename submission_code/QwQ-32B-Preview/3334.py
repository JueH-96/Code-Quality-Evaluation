from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        boxes_used = 0
        current_apples = total_apples
        for box in capacity:
            if current_apples > 0:
                current_apples -= box
                boxes_used += 1
            else:
                break
        return boxes_used