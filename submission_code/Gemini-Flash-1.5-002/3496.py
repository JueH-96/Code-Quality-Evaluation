class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def time_taken(worker_time, reduction):
            return worker_time * (reduction * (reduction + 1)) // 2

        def check(time_limit):
            total_reduction = 0
            for worker_time in workerTimes:
                reduction = 0
                left, right = 0, mountainHeight
                while left <= right:
                    mid = (left + right) // 2
                    if time_taken(worker_time, mid) <= time_limit:
                        reduction = mid
                        left = mid + 1
                    else:
                        right = mid - 1
                total_reduction += reduction
            return total_reduction >= mountainHeight

        left, right = 0, sum(workerTimes) * mountainHeight * (mountainHeight + 1) // 2
        ans = right
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans