class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        n = len(start)
        if n < 2:
            return 0  # According to constraints, n >= 2, so this is just a safeguard
        
        min_start = start[0]
        max_end = start[-1] + d
        low, high = 0, max_end - min_start
        
        while low < high:
            mid = (low + high + 1) // 2
            if self.is_possible(mid, start, d):
                low = mid
            else:
                high = mid - 1
        return low
    
    def is_possible(self, s: int, start: List[int], d: int) -> bool:
        prev = start[0]
        for i in range(1, len(start)):
            current_min = prev + s
            current_start = start[i]
            current_end = current_start + d
            if current_min > current_end:
                return False
            prev = max(current_min, current_start)
        return True