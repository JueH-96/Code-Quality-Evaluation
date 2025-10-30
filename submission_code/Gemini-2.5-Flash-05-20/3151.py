import math
from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        """
        Calculates the minimum time when all tasks have been executed by processors.

        The problem specifies 'n' processors, each with 4 cores, and '4 * n' tasks.
        Each core performs one task independently.
        A processor becomes available at `processorTime[i]`.
        The completion time for a processor is its availability time plus the maximum
        duration of the 4 tasks assigned to its cores. The goal is to minimize the
        overall maximum completion time across all processors.

        Strategy:
        To minimize the overall maximum completion time, we should strategically assign
        tasks to processors. The most effective approach is to pair the processors
        that become available earliest with the longest tasks, and conversely,
        processors that become available latest with the shortest tasks. This helps
        in balancing the workload and preventing any single processor from becoming
        a bottleneck due to both late availability and long tasks.

        1. Sort `processorTime` in ascending order:
           This orders processors from earliest available to latest available.
           `processorTime[0]` is the earliest, `processorTime[n-1]` is the latest.

        2. Sort `tasks` in descending order:
           This orders tasks from longest duration to shortest duration.
           `tasks[0]` is the longest task, `tasks[4n-1]` is the shortest.

        3. Assignment Logic:
           - The earliest available processor (`processorTime[0]`) should handle the
             group of 4 longest tasks (`tasks[0], tasks[1], tasks[2], tasks[3]`).
             Its completion time will be `processorTime[0] + max(tasks[0], ..., tasks[3])`.
             Since `tasks` is sorted descending, `max(tasks[0], ..., tasks[3])` is `tasks[0]`.
           - The second earliest available processor (`processorTime[1]`) should handle
             the next group of 4 longest tasks (`tasks[4], tasks[5], tasks[6], tasks[7]`).
             Its completion time will be `processorTime[1] + tasks[4]`.
           - In general, for the `i`-th processor (at `processorTime[i]`), it will be
             assigned the tasks `tasks[4*i]` through `tasks[4*i + 3]`. The maximum
             duration among these tasks will be `tasks[4*i]`.
             Its completion time will be `processorTime[i] + tasks[4*i]`.

        4. Calculate Overall Maximum:
           The final result is the maximum of all these calculated processor completion times.

        This greedy strategy ensures that the processors which are available later
        are assigned shorter tasks, preventing them from becoming the bottleneck,
        while the tasks with the longest durations are processed by the earliest available
        processors.

        Time Complexity: O(N log N) where N is len(processorTime).
                         Sorting processorTime takes O(N log N).
                         Sorting tasks (which has 4N elements) takes O(4N log(4N)) = O(N log N).
                         The loop runs N times, doing constant work.
        Space Complexity: O(N) due to sorting (e.g., Python's Timsort uses O(N) space in worst case).
        """

        # Step 1: Sort processorTime in ascending order.
        processorTime.sort()

        # Step 2: Sort tasks in descending order.
        tasks.sort(reverse=True)

        max_overall_completion_time = 0
        
        n = len(processorTime)
        
        # Iterate through each processor and assign its corresponding tasks
        for i in range(n):
            # Get the availability time for the current processor
            processor_available_time = processorTime[i]
            
            # Identify the longest task assigned to this processor.
            # Since `tasks` is sorted in descending order, the first task in its
            # assigned block of 4 (tasks[4*i]) will be the longest.
            max_task_duration_for_this_processor = tasks[4 * i]
            
            # Calculate the completion time for this specific processor
            current_processor_completion_time = processor_available_time + max_task_duration_for_this_processor
            
            # Update the overall maximum completion time. This will be our answer.
            max_overall_completion_time = max(max_overall_completion_time, current_processor_completion_time)
            
        return max_overall_completion_time