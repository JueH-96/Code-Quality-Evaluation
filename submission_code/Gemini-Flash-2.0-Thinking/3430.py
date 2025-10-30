class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days

        meetings.sort(key=lambda x: x[0])

        merged_meetings = []
        merged_meetings.append(meetings[0])

        for i in range(1, len(meetings)):
            current_meeting = meetings[i]
            last_merged_meeting = merged_meetings[-1]

            if current_meeting[0] <= last_merged_meeting[1]:
                merged_meetings[-1][1] = max(last_merged_meeting[1], current_meeting[1])
            else:
                merged_meetings.append(current_meeting)

        busy_days_count = 0
        for meeting in merged_meetings:
            busy_days_count += (meeting[1] - meeting[0] + 1)

        available_days = days - busy_days_count
        return available_days