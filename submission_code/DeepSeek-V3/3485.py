class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        n = len(start)
        low = 0
        high = start[-1] + d - start[0]
        
        def is_possible(mid):
            prev = start[0]
            for i in range(1, n):
                current = max(start[i], prev + mid)
                if current > start[i] + d:
                    return False
                prev = current
            return True
        
        result = 0
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                result = mid
                low = mid + 1
            else:
                high = mid - 1
        return result