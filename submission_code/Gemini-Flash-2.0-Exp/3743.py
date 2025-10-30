class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        intervals = sorted(zip(startTime, endTime))
        
        gaps = []
        for i in range(1, n):
            gaps.append(intervals[i][0] - intervals[i-1][1])
        
        gaps.sort(reverse=True)
        
        reduction = 0
        for i in range(min(k, len(gaps))):
            reduction += gaps[i]
        
        total_occupied = intervals[-1][1] - intervals[0][0]
        for i in range(1, n):
            total_occupied -= max(0, intervals[i][0] - intervals[i-1][1])
            
        free_time = eventTime - total_occupied
        
        return min(eventTime, free_time + reduction)