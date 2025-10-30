from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        sorted_cap = sorted(capacity, reverse=True)
        sum_cap = 0
        count = 0
        for c in sorted_cap:
            sum_cap += c
            count += 1
            if sum_cap >= total_apples:
                return count
        return count  # This line is theoretically unreachable due to problem constraints