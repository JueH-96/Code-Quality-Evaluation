from typing import List

class Solution:
  def countDays(self, days: int, meetings: List[List[int]]) -> int:
    if not meetings:
      return days

    # Sort meetings by their start day. This modifies the 'meetings' list in-place.
    meetings.sort(key=lambda x: x[0])

    # List to store the merged intervals
    merged_intervals = []
    
    # Initialize the first active interval being formed.
    # It starts with the details of the first meeting in the sorted list.
    active_interval_start = meetings[0][0]
    active_interval_end = meetings[0][1]

    # Iterate through the rest of the meetings (starting from the second one)
    # to merge them with the active_interval or start new ones.
    for i in range(1, len(meetings)):
        next_meeting_start = meetings[i][0]
        next_meeting_end = meetings[i][1]
        
        # Check if the next meeting overlaps with or is adjacent to the current active_interval.
        # Overlap: next_meeting_start <= active_interval_end
        # Adjacent: next_meeting_start == active_interval_end + 1
        # Combined condition for merging: next_meeting_start <= active_interval_end + 1
        if next_meeting_start <= active_interval_end + 1:
            # The next meeting extends the current active_interval.
            # Update the end of the active_interval if the next meeting ends later.
            active_interval_end = max(active_interval_end, next_meeting_end)
        else:
            # The next meeting does not overlap and is not adjacent.
            # This means the current active_interval is finalized. Add it to merged_intervals.
            merged_intervals.append([active_interval_start, active_interval_end])
            # Start a new active_interval with the details of the current meeting.
            active_interval_start = next_meeting_start
            active_interval_end = next_meeting_end
    
    # After the loop finishes, the last active_interval (being processed or just started)
    # must be added to the list of merged_intervals.
    merged_intervals.append([active_interval_start, active_interval_end])

    # Calculate the total number of days covered by meetings from the merged_intervals.
    total_busy_days = 0
    for start, end in merged_intervals:
        # The number of days in an inclusive interval [start, end] is (end - start + 1).
        total_busy_days += (end - start + 1)
        
    # The count of free days is the total number of available days minus the total_busy_days.
    return days - total_busy_days