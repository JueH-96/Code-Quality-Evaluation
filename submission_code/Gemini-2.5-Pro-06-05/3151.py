from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        """
        Calculates the minimum time when all tasks have been executed by processors.

        The problem is to assign 4 tasks to each of the n processors to minimize the
        overall completion time. The completion time for a single processor is its
        available time plus the time of the longest task assigned to it, since its
        4 cores work in parallel. The overall completion time is the maximum of
        these individual processor completion times.

        This problem can be solved with a greedy approach. To minimize the maximum
        completion time, we should pair the processors that become available late
        (potential bottlenecks) with the shortest tasks. Conversely, processors
        that are available early can be assigned longer tasks.

        The optimal strategy is:
        1. Sort `processorTime` in ascending order to get processors from earliest
           to latest available.
        2. Sort `tasks` in descending order to get tasks from longest to shortest.
        3. Pair the i-th earliest processor with the i-th group of 4 longest tasks.
           The completion time for this processor will be its available time plus the
           longest task in its group. Since the tasks are sorted descending, the
           longest task for the i-th processor will be at index `4*i`.
        4. The final answer is the maximum of these completion times over all processors.
        """
        
        # Sort processor available times in ascending order.
        processorTime.sort()
        
        # Sort task durations in descending order.
        tasks.sort(reverse=True)
        
        n = len(processorTime)
        max_completion_time = 0
        
        # Iterate through each processor and its assigned group of tasks.
        for i in range(n):
            # The i-th earliest processor is processorTime[i].
            # It is assigned the i-th group of 4 longest tasks. These tasks are
            # at indices 4*i, 4*i+1, 4*i+2, 4*i+3 in the reverse-sorted tasks list.
            # The longest task in this group is tasks[4*i].
            current_completion_time = processorTime[i] + tasks[4 * i]
            
            # The overall time is the maximum of all processor completion times.
            max_completion_time = max(max_completion_time, current_completion_time)
                
        return max_completion_time