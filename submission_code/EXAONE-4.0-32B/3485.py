class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        n = len(start)
        high_bound = max(start) + d - min(start)
        
        def feasible(x):
            last = -10**18
            for s in start:
                candidate = max(s, last + x)
                if candidate > s + d:
                    return False
                last = candidate
            return True
        
        low, high = 0, high_bound
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            if feasible(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans