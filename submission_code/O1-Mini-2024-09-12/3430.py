class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days

        # Sort the meetings by start day
        meetings.sort(key=lambda x: x[0])

        merged = []
        current_start, current_end = meetings[0]

        for start, end in meetings[1:]:
            if start > current_end + 1:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        merged.append((current_start, current_end))

        # Calculate total days covered by meetings
        total_covered = 0
        for start, end in merged:
            total_covered += end - start + 1

        # Days without meetings
        return days - total_covered