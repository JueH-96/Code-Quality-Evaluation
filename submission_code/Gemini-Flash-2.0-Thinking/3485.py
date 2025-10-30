class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        n = len(start)
        intervals = sorted([(start[i], start[i] + d) for i in range(n)])

        def is_achievable(score):
            chosen = [0] * n
            chosen[0] = intervals[0][0]
            for i in range(1, n):
                lower_bound = chosen[i - 1] + score
                choice = max(intervals[i][0], lower_bound)
                if choice > intervals[i][1]:
                    return False
                chosen[i] = choice
            return True

        low = 0
        high = 2 * 10**9  # A sufficiently large upper bound
        ans = 0

        while low <= high:
            mid = (low + high) // 2
            if is_achievable(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1

        return ans