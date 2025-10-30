class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        intervals = sorted((s, s + d) for s in start)
        n = len(intervals)
        low = 0
        high = 10**18
        ans = 0
        
        while low <= high:
            mid = (low + high) // 2
            possible = True
            prev = -10**18  # A very small number to start with
            for s, e in intervals:
                desired_min = prev + mid
                current = max(desired_min, s)
                if current > e:
                    possible = False
                    break
                prev = current
            if possible:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans