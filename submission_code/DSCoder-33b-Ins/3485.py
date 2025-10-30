class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        n = len(start)
        max_score = 0
        for i in range(n - 1):
            score = min(start[i + 1] - start[i], d)
            max_score = max(max_score, score)
        return max_score