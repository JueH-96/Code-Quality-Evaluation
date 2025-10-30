class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        res = float('inf')
        for i in range(1, len(start)):
            res = min(res, start[i] - start[i - 1])
        return min(res, d)