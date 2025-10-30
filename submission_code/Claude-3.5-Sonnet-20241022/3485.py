class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        n = len(start)
        intervals = []
        for i in range(n):
            intervals.append([start[i], start[i] + d])
            
        def check(diff):
            prev = float('-inf')
            used = 0
            for left, right in sorted(intervals):
                curr = max(left, prev + diff)
                if curr > right:
                    return False
                prev = curr
                used += 1
            return used == n
            
        left, right = 0, 10**10
        while left < right:
            mid = (left + right + 1) // 2
            if check(mid):
                left = mid
            else:
                right = mid - 1
                
        return left