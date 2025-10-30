class Solution:
    def countDays(self, days: int, meetings: [[int]]) -> int:
        meeting_days = [0] * (days + 1)
        for meeting in meetings:
            meeting_days[meeting[0] - 1:meeting[1]] = [1] * (meeting[1] - meeting[0] + 1)
        return meeting_days.count(0)