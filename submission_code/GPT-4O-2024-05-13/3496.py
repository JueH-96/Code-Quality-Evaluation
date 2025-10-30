class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def time_to_reduce_height(worker_time, height):
            return worker_time * (height * (height + 1)) // 2
        
        left, right = 1, mountainHeight * max(workerTimes) * (mountainHeight + 1) // 2
        
        while left < right:
            mid = (left + right) // 2
            total_height_reduced = 0
            
            for worker_time in workerTimes:
                low, high = 0, mountainHeight
                while low < high:
                    mid_height = (low + high + 1) // 2
                    if time_to_reduce_height(worker_time, mid_height) <= mid:
                        low = mid_height
                    else:
                        high = mid_height - 1
                
                total_height_reduced += low
                if total_height_reduced >= mountainHeight:
                    break
            
            if total_height_reduced >= mountainHeight:
                right = mid
            else:
                left = mid + 1
        
        return left