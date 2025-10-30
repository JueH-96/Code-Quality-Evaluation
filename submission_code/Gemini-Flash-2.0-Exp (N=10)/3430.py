class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        available_days = set(range(1, days + 1))
        for start, end in meetings:
            for day in range(start, end + 1):
                if day in available_days:
                    available_days.remove(day)
        return len(available_days)