class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        booked_days = set()
        for meeting in meetings:
            start_day, end_day = meeting
            for day in range(start_day, end_day + 1):
                booked_days.add(day)
        return days - len(booked_days)