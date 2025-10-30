class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        # Calculate durations of each meeting
        durations = [endTime[i] - startTime[i] for i in range(n)]
        
        def can_achieve_gap(gap):
            # Try to achieve a gap of size 'gap' by moving at most k meetings
            moves_needed = 0
            curr_time = 0
            
            for i in range(n):
                # If there's a gap larger than target gap, we need to move meetings
                if startTime[i] - curr_time >= gap:
                    curr_time = endTime[i]
                else:
                    # Need to move this meeting to maintain gap
                    moves_needed += 1
                    curr_time = curr_time + durations[i]
                
                # Check if we exceed event time
                if curr_time > eventTime:
                    return False
            
            return moves_needed <= k
        
        # Binary search for the maximum possible gap
        left, right = 0, eventTime
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            if can_achieve_gap(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result