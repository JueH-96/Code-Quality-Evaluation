from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        total_sum = sum(2 * x - 1 for x in possible)
        current_sum = 0
        for i in range(len(possible) - 1):
            current_sum += 2 * possible[i] - 1
            if current_sum > total_sum - current_sum:
                return i + 1
        return -1