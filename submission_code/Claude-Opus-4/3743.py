class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        meetings = list(zip(startTime, endTime))
        
        def maxGapAfterMoving(gap_start, gap_end, left_meetings, right_meetings):
            # Try to maximize the gap [gap_start, gap_end] by moving meetings
            # left_meetings: meetings to the left that can be moved left
            # right_meetings: meetings to the right that can be moved right
            
            # Calculate how much we can extend the gap
            left_extension = 0
            right_extension = 0
            moves_used = 0
            
            # Try moving meetings on the left further left
            left_pos = gap_start
            for i in range(len(left_meetings) - 1, -1, -1):
                if moves_used >= k:
                    break
                start, end = left_meetings[i]
                duration = end - start
                
                # Find the leftmost position for this meeting
                if i > 0:
                    prev_end = left_meetings[i-1][1]
                    new_start = max(prev_end, left_pos - duration)
                else:
                    new_start = max(0, left_pos - duration)
                
                if new_start < start:
                    left_extension += start - new_start
                    left_pos = new_start
                    moves_used += 1
                else:
                    left_pos = start
            
            # Try moving meetings on the right further right
            right_pos = gap_end
            for i in range(len(right_meetings)):
                if moves_used >= k:
                    break
                start, end = right_meetings[i]
                duration = end - start
                
                # Find the rightmost position for this meeting
                if i < len(right_meetings) - 1:
                    next_start = right_meetings[i+1][0]
                    new_end = min(next_start, right_pos + duration)
                else:
                    new_end = min(eventTime, right_pos + duration)
                
                new_start = new_end - duration
                if new_start > start:
                    right_extension += new_start - start
                    right_pos = new_end
                    moves_used += 1
                else:
                    right_pos = end
            
            return gap_end - gap_start + left_extension + right_extension
        
        max_free = 0
        
        # Check gap before first meeting
        if meetings[0][0] > 0:
            gap = meetings[0][0]
            # Can potentially move first k meetings to the right
            right_meetings = meetings[:min(k, n)]
            extended_gap = maxGapAfterMoving(0, meetings[0][0], [], right_meetings)
            max_free = max(max_free, extended_gap)
        
        # Check gaps between meetings
        for i in range(n - 1):
            gap_start = meetings[i][1]
            gap_end = meetings[i + 1][0]
            if gap_end > gap_start:
                # Meetings that can be moved left (up to k meetings before the gap)
                left_start = max(0, i - k + 1)
                left_meetings = meetings[left_start:i + 1]
                
                # Meetings that can be moved right (up to k meetings after the gap)
                right_end = min(n, i + 1 + k)
                right_meetings = meetings[i + 1:right_end]
                
                extended_gap = maxGapAfterMoving(gap_start, gap_end, left_meetings, right_meetings)
                max_free = max(max_free, extended_gap)
        
        # Check gap after last meeting
        if meetings[-1][1] < eventTime:
            gap = eventTime - meetings[-1][1]
            # Can potentially move last k meetings to the left
            left_meetings = meetings[max(0, n - k):]
            extended_gap = maxGapAfterMoving(meetings[-1][1], eventTime, left_meetings, [])
            max_free = max(max_free, extended_gap)
        
        return max_free