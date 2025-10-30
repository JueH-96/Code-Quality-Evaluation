class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start = sorted(start)
        low = 0
        high = start[-1] + d - start[0]
        
        def is_possible(m):
            prev = -float('inf')
            for i in range(len(start)):
                current_low = start[i]
                current_high = current_low + d
                x_i = max(prev + m, current_low)
                if x_i > current_high:
                    return False
                prev = x_i
            return True
        
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                low = mid + 1
            else:
                high = mid - 1
        return high