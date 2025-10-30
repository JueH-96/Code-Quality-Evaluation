class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: list[int], endTime: list[int]) -> int:
        n = len(startTime)
        if n == 0:
            return eventTime
        
        dp = [float('inf')] * (k + 1)
        dp[0] = 0
        max_free = 0
        
        for i in range(n):
            new_dp = [float('inf')] * (k + 1)
            s = startTime[i]
            e = endTime[i]
            
            for j in range(k, -1, -1):
                if dp[j] == float('inf'):
                    continue
                
                # Option 1: Don't reschedule the current meeting
                if s >= dp[j]:
                    if new_dp[j] > e:
                        new_dp[j] = e
                    current_gap = s - dp[j]
                    if current_gap > max_free:
                        max_free = current_gap
                
                # Option 2: Reschedule the current meeting
                if j < k:
                    new_end = dp[j] + (e - s)
                    if new_end <= eventTime:
                        if new_dp[j + 1] > new_end:
                            new_dp[j + 1] = new_end
            
            dp = new_dp
        
        # Check the gap after the last meeting
        if dp[k] != float('inf'):
            gap_after = eventTime - dp[k]
            if gap_after > max_free:
                max_free = gap_after
        
        # Check the gap before the first meeting (original start time)
        gap_before = startTime[0]
        if gap_before > max_free:
            max_free = gap_before
        
        return max_free