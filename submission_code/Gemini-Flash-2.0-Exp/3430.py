class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        scheduled = set()
        for meeting in meetings:
            for i in range(meeting[0], meeting[1] + 1):
                scheduled.add(i)
        return days - len(scheduled)