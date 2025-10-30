from collections import Counter

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        hour_count = Counter(hours)
        complete_days = 0
        
        for hour, count in hour_count.items():
            for multiple in range(1, 25):
                complement = 24 * multiple
                if complement >= hour and complement - hour in hour_count:
                    if complement == hour:
                        complete_days += count * (count - 1) // 2
                    else:
                        complete_days += count * hour_count[complement]
        
        return complete_days