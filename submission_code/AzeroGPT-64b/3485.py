class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        min_diff = start[1] - start[0]
        max_num = start[0] + d
        for i in range(1, len(start)):
            max_num = max(max_num, start[i] + d)
            min_diff = min(min_diff, max_num - start[i])
        return min_diff