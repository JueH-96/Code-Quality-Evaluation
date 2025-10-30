from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # We have n processors (each with 4 cores) and 4*n tasks.
        # Each processor i, which becomes available at time p, is given 4 tasks.
        # Since each core works independently (and in parallel),
        # the finish time for processor i is p + (max time among its four tasks).
        # Our goal is to minimize the overall time when every processor has finished,
        # i.e. minimize max_i (p_i + max(task times assigned to i)).
        #
        # Notice that only the maximum task duration in a processor's batch actually affects 
        # its finish time. Therefore, when partitioning the tasks into n groups of 4 tasks each,
        # it is optimal to “control” the group maximums.
        # In particular, one can show (via an exchange argument or rearrangement inequality)
        # that an optimal assignment is achieved when we partition the tasks (after sorting ascending)
        # into n consecutive groups of 4. Let M_i be the maximum of the i-th group.
        # Then, to “balance” the sums p[i] + M_i, we want to assign the smallest available M_i
        # to the processor whose available time is largest.
        #
        # Thus, the strategy is:
        # 1. Sort processorTime in descending order (largest available time first).
        # 2. Sort tasks in ascending order.
        # 3. Partition the tasks into n groups (each of 4 consecutive tasks in sorted order).
        #    In each group the maximum is the 4th task.
        # 4. Pair the groups with processors so that:
        #       finish time = processorTime[i] + tasks[4*i + 3]   for i = 0, 1, ..., n-1,
        #    where processorTime[i] is the i-th largest available time
        #    and tasks[4*i + 3] is the maximum of the i-th (smallest) group.
        # 5. The answer is the maximum finish time among all processors.
        
        n = len(processorTime)
        processorTime.sort(reverse=True)
        tasks.sort()
        
        answer = 0
        # For each processor (i from 0 to n-1) assign the group of tasks with the
        # 4 consecutive smallest tasks (from the sorted list); the 4th element in the
        # group gives the maximum running time for that processor.
        for i in range(n):
            finish_time = processorTime[i] + tasks[4 * i + 3]
            answer = max(answer, finish_time)
            
        return answer