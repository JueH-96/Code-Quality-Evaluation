class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        meetings.sort()
        merged_meetings = []
        for meeting in meetings:
            if not merged_meetings or meeting[0] > merged_meetings[-1][1]:
                merged_meetings.append(meeting)
            else:
                merged_meetings[-1][1] = max(merged_meetings[-1][1], meeting[1])

        count = 0
        last_day = 0
        for start, end in merged_meetings:
            count += max(0, start - last_day - 1)
            last_day = max(last_day, end)

        count += max(0, days - last_day)
        return count