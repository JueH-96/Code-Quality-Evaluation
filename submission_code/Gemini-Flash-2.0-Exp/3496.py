class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def check(time):
            heightReduced = 0
            for workerTime in workerTimes:
                l, r = 0, mountainHeight
                ans = 0
                while l <= r:
                    mid = (l + r) // 2
                    cost = workerTime * mid * (mid + 1) // 2
                    if cost <= time:
                        ans = mid
                        l = mid + 1
                    else:
                        r = mid - 1
                heightReduced += ans
            return heightReduced >= mountainHeight

        l, r = 0, sum(workerTimes) * mountainHeight
        ans = r
        while l <= r:
            mid = (l + r) // 2
            if check(mid):
                ans = mid
                r = mid - 1
            else:
                l = mid + 1
        return ans