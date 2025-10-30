from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        
        used_capacity = 0
        for i, cap in enumerate(capacity):
            used_capacity += cap
            if used_capacity >= total_apples:
                return i + 1