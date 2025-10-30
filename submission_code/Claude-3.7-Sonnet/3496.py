class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def can_reduce_to_zero(time_limit):
            total_reduced = 0
            for worker_time in workerTimes:
                # Formula to find maximum height a worker can reduce in given time:
                # This solves for x in: worker_time*(1+2+...+x) ≤ time_limit
                # Using the formula for sum of first x integers: x*(x+1)/2
                max_height = int((-1 + (1 + 8 * (time_limit / worker_time))**0.5) / 2)
                total_reduced += max_height
                if total_reduced >= mountainHeight:
                    return True
            return False
        
        # Binary search for the minimum time
        left = 1
        right = max(workerTimes) * mountainHeight * (mountainHeight + 1) // 2
        
        while left < right:
            mid = (left + right) // 2
            if can_reduce_to_zero(mid):
                right = mid
            else:
                left = mid + 1
        
        return left