from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        n = len(start)
        intervals = [(s, s + d) for s in start]
        intervals.sort()
        
        def is_valid(score):
            count = 0
            last_chosen = float('-inf')
            for low, high in intervals:
                if low >= last_chosen + score:
                    count += 1
                    last_chosen = high
                    if count == n:
                        return True
            return False
        
        left, right = 0, intervals[-1][1] - intervals[0][0]
        while left < right:
            mid = (left + right + 1) // 2
            if is_valid(mid):
                left = mid
            else:
                right = mid - 1
        
        return left