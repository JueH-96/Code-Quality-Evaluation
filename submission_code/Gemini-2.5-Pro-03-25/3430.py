# Import List from typing
from typing import List

class Solution:
    """
    Calculates the number of days an employee is available for work but has no meetings scheduled.

    The core idea is to determine the total number of unique days that have at least one meeting scheduled.
    This is achieved by merging overlapping meeting intervals and then summing the lengths of these merged intervals.
    Finally, the total number of busy days is subtracted from the total available days to find the count of free days.

    The approach involves:
    1. Sorting the meeting intervals based on their start times. This is crucial for efficiently merging overlapping intervals.
    2. Merging the sorted intervals. We iterate through the sorted intervals and combine any that overlap or are adjacent into a single, larger interval.
    3. Calculating the total number of days covered by these merged intervals (total busy days).
    4. Subtracting the total busy days from the total number of available work days (`days`) to find the count of days with no meetings.
    """
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        """
        Args:
            days: An integer representing the total number of available work days (numbered 1 to `days`).
            meetings: A list of lists, where each inner list `[start_i, end_i]` represents a meeting
                      scheduled from `start_i` to `end_i` (inclusive).

        Returns:
            An integer representing the count of days when the employee is available and has no meetings scheduled.
        """

        # If the meetings list is empty, the employee is free on all 'days'.
        if not meetings:
            return days

        # Sort the meetings based on their start day.
        # This step is necessary to easily merge overlapping or adjacent intervals.
        # Time complexity: O(N log N), where N is the number of meetings.
        # Python's Timsort is efficient and has O(N) space complexity in the worst case.
        meetings.sort(key=lambda x: x[0])

        # Merge overlapping intervals.
        # We use a list `merged` to store the non-overlapping, consolidated intervals.
        # Space complexity: O(M) or O(N) in the worst case for the merged list,
        # where M is the number of merged intervals (M <= N).
        merged = []
        for start, end in meetings:
            # Check if the `merged` list is empty or if the current meeting starts
            # strictly after the last merged interval ends.
            # `start > merged[-1][1]` indicates a gap between the last merged interval and the current one.
            if not merged or start > merged[-1][1]:
                # If there's no overlap, add the current meeting as a new interval to `merged`.
                merged.append([start, end])
            else:
                # If there is an overlap (or they are adjacent), merge the current meeting
                # with the last interval in `merged` by potentially extending the end day.
                # The end day of the merged interval becomes the maximum of its current end
                # and the end day of the current meeting.
                merged[-1][1] = max(merged[-1][1], end)

        # Calculate the total number of days occupied by meetings.
        # Iterate through the `merged` list, which contains the consolidated busy periods.
        # Time complexity: O(M), where M is the number of merged intervals.
        total_busy_days = 0
        for start, end in merged:
            # The number of days in an interval [start, end] (inclusive) is calculated as (end - start + 1).
            # According to constraints, 1 <= start <= end <= days.
            total_busy_days += (end - start + 1)

        # The number of free days is the total number of available days minus the total busy days.
        # Python integers handle large numbers (up to 10^9 for `days`), so overflow is not an issue.
        free_days = days - total_busy_days

        # The constraints guarantee that all meeting intervals are within the range [1, days].
        # Therefore, `total_busy_days` cannot exceed `days`, ensuring `free_days` is non-negative.
        return free_days