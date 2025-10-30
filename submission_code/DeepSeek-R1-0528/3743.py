class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        total_dur = 0
        for i in range(n):
            total_dur += (endTime[i] - startTime[i])
        
        def canHaveFreeTime(x):
            if x == 0:
                return True
            if eventTime - total_dur < x:
                return False
            max_gap = startTime[0]
            for i in range(n - 1):
                gap = startTime[i+1] - endTime[i]
                if gap > max_gap:
                    max_gap = gap
            last_gap = eventTime - endTime[n-1]
            if last_gap > max_gap:
                max_gap = last_gap
            if max_gap >= x:
                return True
            
            r = 0
            for l in range(n):
                if r < l:
                    r = l
                while r < n:
                    left_bound = endTime[l-1] if l > 0 else 0
                    if r < n - 1:
                        right_bound = startTime[r+1]
                    else:
                        right_bound = eventTime
                    gap_size = right_bound - left_bound
                    if gap_size >= x:
                        break
                    else:
                        r += 1
                if r < n and (r - l + 1) <= k:
                    return True
            return False
        
        lo, hi = 0, eventTime
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if canHaveFreeTime(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo