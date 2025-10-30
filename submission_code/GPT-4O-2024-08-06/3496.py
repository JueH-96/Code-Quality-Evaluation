class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def time_to_reduce_height(worker_time, height):
            # Calculate the time taken for a worker to reduce the mountain by `height`
            return sum(worker_time * i for i in range(1, height + 1))
        
        left, right = 0, max(workerTimes) * mountainHeight * (mountainHeight + 1) // 2
        
        while left < right:
            mid = (left + right) // 2
            total_height_reduced = 0
            
            for worker_time in workerTimes:
                # Calculate the maximum height this worker can reduce in `mid` seconds
                low, high = 0, mountainHeight
                while low < high:
                    mid_height = (low + high + 1) // 2
                    if time_to_reduce_height(worker_time, mid_height) <= mid:
                        low = mid_height
                    else:
                        high = mid_height - 1
                total_height_reduced += low
            
            if total_height_reduced >= mountainHeight:
                right = mid
            else:
                left = mid + 1
        
        return left