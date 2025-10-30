class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        if not meetings:
            return days  # No meetings scheduled
        
        # Sort the meetings based on start day
        meetings.sort()
        merged = []
        for interval in meetings:
            if not merged or interval[0] > merged[-1][1]:
                # No overlap
                merged.append(interval)
            else:
                # Overlap, merge
                merged[-1][1] = max(merged[-1][1], interval[1])
        
        total_covered = 0
        for interval in merged:
            total_covered += interval[1] - interval[0] + 1
        return days - total_covered