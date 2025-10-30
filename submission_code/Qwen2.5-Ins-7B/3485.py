class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        min_diff = float('inf')
        for i in range(len(start)):
            for j in range(i+1, len(start)):
                min_diff = min(min_diff, abs(start[i] - start[j]))
        return min_diff