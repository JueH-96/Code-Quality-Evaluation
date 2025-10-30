class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        counts = [0] * 24
        res = 0
        for h in hours:
            rem = h % 24
            complement = (24 - rem) % 24
            res += counts[complement]
            counts[rem] += 1
        return res