class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        intervals = sorted((s, s + d) for s in start)
        if not intervals:
            return 0
        
        min_start = intervals[0][0]
        max_end = max(e for s, e in intervals)
        
        low = 0
        high = max_end - min_start
        best = 0
        
        while low <= high:
            mid = (low + high) // 2
            last = -float('inf')
            possible = True
            
            for s, e in intervals:
                earliest = max(s, last + mid)
                if earliest > e:
                    possible = False
                    break
                last = earliest
            
            if possible:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return best