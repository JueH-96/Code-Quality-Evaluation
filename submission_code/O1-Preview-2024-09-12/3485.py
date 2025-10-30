class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        n = len(start)
        start.sort()
        left = 0
        right = 10**18  # Upper bound for binary search
        def can_place(m):
            prev = None
            for i in range(n):
                if prev is None:
                    x_i = start[i]
                else:
                    x_i = max(start[i], prev + m)
                if x_i > start[i] + d:
                    return False
                prev = x_i
            return True
        while left < right:
            mid = (left + right + 1) // 2
            if can_place(mid):
                left = mid
            else:
                right = mid - 1
        return left