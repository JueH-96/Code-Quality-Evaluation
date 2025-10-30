class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        
        def can_achieve_free_time(free_length):
            # Try each possible starting point for the free time
            for start in range(eventTime - free_length + 1):
                end = start + free_length
                meetings_to_move = 0
                
                # Count how many meetings overlap with this interval
                for i in range(n):
                    if startTime[i] < end and endTime[i] > start:
                        meetings_to_move += 1
                
                # If we need to move at most k meetings, this free time is achievable
                if meetings_to_move <= k:
                    return True
            
            return False
        
        # Binary search for the maximum free time length
        left, right = 0, eventTime
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            if can_achieve_free_time(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result