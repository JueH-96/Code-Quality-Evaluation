class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        from collections import Counter
        hours = [hour % 24 for hour in hours]
        counter = Counter(hours)
        pairs = 0
        for hour, count in counter.items():
            if count > 1 and hour == 0:
                pairs += count * (count - 1) // 2
            elif 24 - hour in counter:
                pairs += count * counter[24 - hour]
                counter[24 - hour] = 0
        return pairs