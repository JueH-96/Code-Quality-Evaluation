class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def check(seconds):
            total_height = 0
            for time in workerTimes:
                height = 0
                l, r = 0, mountainHeight
                while l <= r:
                    mid = (l + r) // 2
                    if time * mid * (mid + 1) // 2 <= seconds:
                        height = mid
                        l = mid + 1
                    else:
                        r = mid - 1
                total_height += height
            return total_height >= mountainHeight

        left, right = 1, mountainHeight * (mountainHeight + 1) // 2 * max(workerTimes)
        ans = right
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans