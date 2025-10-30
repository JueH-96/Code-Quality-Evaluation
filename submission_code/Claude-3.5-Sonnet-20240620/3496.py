class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def time_to_reduce(worker_time, height):
            return sum(worker_time * i for i in range(1, height + 1))

        def can_reduce_in_time(time):
            total_reduction = sum(int(((1 + 8 * time // t) ** 0.5 - 1) // 2) for t in workerTimes)
            return total_reduction >= mountainHeight

        left, right = 0, mountainHeight * max(workerTimes)
        while left < right:
            mid = (left + right) // 2
            if can_reduce_in_time(mid):
                right = mid
            else:
                left = mid + 1

        return left