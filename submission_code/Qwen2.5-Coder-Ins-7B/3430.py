class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        meetings.sort()
        end = 0
        count = 0
        for start, end in meetings:
            if start > end:
                count += start - end - 1
            end = max(end, end)
        if days > end:
            count += days - end
        return count