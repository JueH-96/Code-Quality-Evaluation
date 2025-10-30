from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        low, high = 0, start[-1] - start[0]
        while low < high:
            mid = (low + high + 1) // 2
            if self.is_valid(start, d, mid):
                low = mid
            else:
                high = mid - 1
        return low

    def is_valid(self, start: List[int], d: int, min_diff: int) -> bool:
        last_picked = start[0] + d
        for i in range(1, len(start)):
            if start[i] + d >= last_picked + min_diff:
                last_picked = start[i] + d
            elif start[i] >= last_picked + min_diff:
                last_picked = start[i]
            else:
                continue
        return True