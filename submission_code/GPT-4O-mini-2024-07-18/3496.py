from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def time_to_reduce(worker_time, x):
            return worker_time * (x * (x + 1)) // 2
        
        from itertools import combinations
        
        min_time = float('inf')
        
        for i in range(1, len(workerTimes) + 1):
            for workers in combinations(workerTimes, i):
                total_time = 0
                remaining_height = mountainHeight
                
                for worker in workers:
                    height_to_reduce = min(remaining_height, len(workers))
                    total_time = max(total_time, time_to_reduce(worker, height_to_reduce))
                    remaining_height -= height_to_reduce
                
                if remaining_height <= 0:
                    min_time = min(min_time, total_time)
        
        return min_time