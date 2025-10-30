class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        n = len(start)
        low, high = 0, start[-1] + d - start[0]
        
        def feasible(k):
            last = -10**18
            for s in start:
                candidate = max(last + k, s)
                if candidate > s + d:
                    return False
                last = candidate
            return True
        
        while low < high:
            mid = (low + high + 1) // 2
            if feasible(mid):
                low = mid
            else:
                high = mid - 1
        return low