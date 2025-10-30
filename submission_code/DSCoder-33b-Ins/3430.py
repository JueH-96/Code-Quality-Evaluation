class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        meeting_days = [0] * (days + 1)
        for meeting in meetings:
            for day in range(meeting[0], meeting[1] + 1):
                meeting_days[day] = 1
        return meeting_days.count(0)