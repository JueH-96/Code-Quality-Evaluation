class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        
        # Calculate current gaps
        gaps = []
        gaps.append(startTime[0])  # Before first meeting
        for i in range(1, n):
            gaps.append(startTime[i] - endTime[i-1])
        gaps.append(eventTime - endTime[-1])  # After last meeting
        
        max_free_time = max(gaps)
        
        # Try removing k consecutive meetings and see max gap we can create
        for i in range(n - k + 1):
            # Remove meetings from i to i+k-1
            left_pos = endTime[i-1] if i > 0 else 0
            right_pos = startTime[i+k] if i+k < n else eventTime
            
            # Total duration of removed meetings
            removed_duration = sum(endTime[j] - startTime[j] for j in range(i, i+k))
            
            # Space available for these meetings
            available_space = right_pos - left_pos
            
            # We can create a gap of (available_space - removed_duration)
            if available_space >= removed_duration:
                possible_gap = available_space - removed_duration
                max_free_time = max(max_free_time, possible_gap)
        
        return max_free_time