from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        meetings = [(startTime[i], endTime[i]) for i in range(n)]
        
        # Calculate the initial free time segments
        free_segments = []
        last_end = 0
        
        for start, end in meetings:
            if last_end < start:
                free_segments.append((last_end, start))
            last_end = end
        
        if last_end < eventTime:
            free_segments.append((last_end, eventTime))
        
        # If we can reschedule k meetings, we need to find the best k meetings to move
        # We will calculate the potential free time we can create by moving meetings
        potential_gaps = []
        
        for i in range(n):
            meeting_duration = endTime[i] - startTime[i]
            # Check if we can move the meeting to the left
            if i == 0:
                # First meeting can only move to the start of the event
                if startTime[i] > 0:
                    potential_gaps.append((0, startTime[i]))
            else:
                # Move to the end of the previous meeting
                prev_end = endTime[i - 1]
                if startTime[i] > prev_end:
                    potential_gaps.append((prev_end, startTime[i]))
            
            # Check if we can move the meeting to the right
            if i == n - 1:
                # Last meeting can only move to the end of the event
                if endTime[i] < eventTime:
                    potential_gaps.append((endTime[i], eventTime))
            else:
                # Move to the start of the next meeting
                next_start = startTime[i + 1]
                if endTime[i] < next_start:
                    potential_gaps.append((endTime[i], next_start))
        
        # Sort potential gaps by their lengths
        potential_gaps.sort(key=lambda x: x[1] - x[0], reverse=True)
        
        # Calculate the maximum free time we can achieve
        max_free_time = 0
        
        # Use the largest k gaps
        for i in range(min(k, len(potential_gaps))):
            max_free_time += potential_gaps[i][1] - potential_gaps[i][0]
        
        # Add the initial free time segments
        for start, end in free_segments:
            max_free_time += end - start
        
        return max_free_time