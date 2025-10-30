import collections
from typing import List

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        # If there are no meetings, all 'days' are free.
        if not meetings:
            return days

        # Sort the meetings by their start day. This is crucial for correctly merging
        # overlapping or adjacent intervals. If start days are the same, the order
        # of end days doesn't significantly affect the final coverage, but Python's
        # default sort will handle it consistently.
        meetings.sort()

        total_covered_days = 0
        
        # Initialize the current merged interval with the first meeting's details.
        # We will then iterate from the second meeting to compare and merge.
        current_merge_start = meetings[0][0]
        current_merge_end = meetings[0][1]

        # Iterate through the meetings starting from the second one (index 1).
        for i in range(1, len(meetings)):
            start, end = meetings[i][0], meetings[i][1]

            # Check if the current meeting overlaps with or is directly adjacent to
            # the `current_merged_interval`.
            # Overlap: `start <= current_merge_end`
            # Adjacency: `start == current_merge_end + 1` (e.g., [1,3] and [4,5] should merge)
            # Both cases are covered by `start <= current_merge_end + 1`.
            if start <= current_merge_end + 1:
                # Merge: Extend the `current_merge_end` to cover the new meeting.
                # We take the maximum of the current `current_merge_end` and the
                # new meeting's `end` because the new meeting might extend further.
                current_merge_end = max(current_merge_end, end)
            else:
                # No overlap or adjacency. The current meeting is completely separate
                # from the `current_merged_interval`.
                # First, add the length of the `current_merged_interval` (which is now finalized)
                # to the `total_covered_days`.
                total_covered_days += (current_merge_end - current_merge_start + 1)
                # Then, start a new `current_merged_interval` with the details of the
                # current meeting `[start, end]`.
                current_merge_start = start
                current_merge_end = end
        
        # After the loop finishes, there will always be one last `current_merged_interval`
        # whose length hasn't been added to `total_covered_days` yet. This interval
        # represents either the very last merged block or the only meeting if all merged
        # into one. Add its length.
        total_covered_days += (current_merge_end - current_merge_start + 1)

        # The final result is the total number of available days minus the sum of days
        # covered by all merged meetings.
        return days - total_covered_days