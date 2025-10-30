from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Prepare horizontal and vertical intervals
        horizontal = [(rect[1], rect[3]) for rect in rectangles]
        vertical = [(rect[0], rect[2]) for rect in rectangles]
        return self.can_split(horizontal) or self.can_split(vertical)
        
    def can_split(self, intervals: List[tuple]) -> bool:
        # Check if there are at least two valid splits
        intervals.sort(key=lambda x: x[1])
        n = len(intervals)
        if n < 2:
            return False
        
        # Compute suffix minimum of start values
        suffix_min = [0] * n
        suffix_min[-1] = intervals[-1][0]
        for i in range(n - 2, -1, -1):
            suffix_min[i] = min(intervals[i][0], suffix_min[i + 1])
        
        count = 0
        for i in range(n - 1):  # i ranges from 0 to n-2 inclusive
            if intervals[i][1] <= suffix_min[i + 1]:
                count += 1
                if count >= 2:
                    return True
        return count >= 2