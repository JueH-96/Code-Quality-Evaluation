class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        mod_hours = [h % 24 for h in hours]
        freq = [0] * 24
        for m in mod_hours:
            freq[m] += 1
        count = 0
        for r in range(24):
            s = (24 - r) % 24
            if s < r:
                continue
            if s == r:
                count += (freq[r] * (freq[r] - 1)) // 2
            else:
                if freq[s] > 0:
                    count += freq[r] * freq[s]
        return count