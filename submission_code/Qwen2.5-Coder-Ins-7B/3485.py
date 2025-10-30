class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        max_score = 0
        for i in range(len(start) - d + 1):
            max_score = max(max_score, start[i + d - 1] - start[i])
        return max_score