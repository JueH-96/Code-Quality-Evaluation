from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        # Sort the worker times in ascending order
        workerTimes.sort()
        
        # Initialize the minimum time
        min_time = 0
        
        # Initialize the index of the current worker
        worker_index = 0
        
        # While the mountain height is greater than 0
        while mountainHeight > 0:
            # Calculate the time taken by the current worker to reduce the mountain height by 1
            time_taken = workerTimes[worker_index]
            
            # Update the minimum time
            min_time = max(min_time, time_taken)
            
            # Reduce the mountain height by 1
            mountainHeight -= 1
            
            # Move to the next worker
            worker_index = (worker_index + 1) % len(workerTimes)
        
        # Return the minimum time
        return min_time