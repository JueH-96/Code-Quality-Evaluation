class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        intervals = sorted([(start[i], start[i]+d) for i in range(len(start))])
        res, cur = 0, 0
        for i, (l, r) in enumerate(intervals):
            if i % 2 == 0:
                cur = max(cur, l)
            else:
                res = max(res, r-cur)
        return res