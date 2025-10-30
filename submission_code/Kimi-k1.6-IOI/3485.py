class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        start.sort()
        n = len(start)
        left = 0
        right = 10**18
        ans = 0
        
        while left <= right:
            mid = (left + right) // 2
            possible = True
            prev = start[0]
            for i in range(1, n):
                required = prev + mid
                current_start = start[i]
                current_end = start[i] + d
                min_x = max(required, current_start)
                if min_x > current_end:
                    possible = False
                    break
                prev = min_x
            if possible:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans