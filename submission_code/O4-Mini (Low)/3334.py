from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        # Sort capacities in descending order to use the biggest boxes first
        capacity.sort(reverse=True)
        
        used = 0
        accumulated = 0
        for cap in capacity:
            accumulated += cap
            used += 1
            if accumulated >= total_apples:
                return used
        
        # Problem guarantees redistribution is possible, so we never actually reach here
        return used