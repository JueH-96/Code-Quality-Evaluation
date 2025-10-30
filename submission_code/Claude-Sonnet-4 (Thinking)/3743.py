class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        
        # Collect all critical time points where optimal segments might start/end
        time_points = set([0, eventTime])
        for i in range(n):
            time_points.add(startTime[i])
            time_points.add(endTime[i])
        
        time_points = sorted(time_points)
        max_free_time = 0
        
        # Try all possible segments [L, R]
        for i in range(len(time_points)):
            for j in range(i, len(time_points)):
                L, R = time_points[i], time_points[j]
                
                # Count meetings that overlap with segment [L, R]
                overlapping_meetings = 0
                for m in range(n):
                    # Meeting [startTime[m], endTime[m]] overlaps with [L, R] if:
                    # startTime[m] < R and endTime[m] > L
                    if startTime[m] < R and endTime[m] > L:
                        overlapping_meetings += 1
                
                # If we can move all overlapping meetings (≤ k), this segment is achievable
                if overlapping_meetings <= k:
                    max_free_time = max(max_free_time, R - L)
        
        return max_free_time