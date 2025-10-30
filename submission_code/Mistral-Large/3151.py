from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort both processorTime and tasks
        processorTime.sort()
        tasks.sort()

        # Initialize the minimum time to 0
        min_time = 0

        # Iterate over the processors
        for i in range(len(processorTime)):
            # Calculate the time taken by the current processor to finish its tasks
            time_taken = processorTime[i] + tasks[-(4 * (i + 1))]
            # Update the minimum time
            min_time = max(min_time, time_taken)

        return min_time