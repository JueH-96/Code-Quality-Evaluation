class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        occupied_days = set()
        for meeting in meetings:
            start, end = meeting
            for day in range(start, end + 1):
                occupied_days.add(day)
        return days - len(occupied_days)