class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # Create a set to keep track of days with meetings
        meeting_days = set()

        # Iterate through each meeting and mark the days with meetings
        for start, end in meetings:
            for day in range(start, end + 1):
                meeting_days.add(day)

        # Count the days without meetings
        available_days = days - len(meeting_days)

        return available_days