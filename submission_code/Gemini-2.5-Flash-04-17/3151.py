from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        """
        Calculates the minimum time when all tasks have been executed.

        We have n processors, each with 4 cores. Total tasks = 4*n.
        Each core executes one task.
        ProcessorTime[i] is when processor i becomes available.
        tasks[j] is the execution time of task j.
        A task j on a core of processor i finishes at processorTime[i] + tasks[j].
        A processor finishes when all its 4 assigned tasks are done, which is
        max(processorTime[i] + task_time for the 4 assigned tasks)
        = processorTime[i] + max(task_time for the 4 assigned tasks).

        We want to minimize the maximum finish time across all processors.

        Intuitively, to minimize the overall maximum completion time,
        we should pair the processors that become available earliest
        with the tasks that take the longest, and vice versa. This helps
        balance the load and prevent one processor from finishing excessively late.

        Algorithm:
        1. Sort processorTime in ascending order.
           processorTime[0] is the earliest available processor.
           processorTime[n-1] is the latest available processor.
        2. Sort tasks in ascending order.
           tasks[0] is the shortest task.
           tasks[4n-1] is the longest task.
        3. Pair the i-th earliest processor (processorTime[i]) with the
           i-th group of 4 longest tasks.
           The longest 4 tasks are tasks[4n-4]...tasks[4n-1].
           The next longest 4 tasks are tasks[4n-8]...tasks[4n-5].
           The i-th group of 4 longest tasks (0-indexed i) corresponds
           to tasks from index 4n - 4*(i+1) up to 4n - 4*i - 1.
           The maximum task time in this i-th group is at index 4n - 4*i - 1.
           So, processorTime[i] is paired with tasks[4n - 4*i - 1].
        4. Calculate the finish time for each processor `i` as
           `processorTime[i] + tasks[4n - 4*i - 1]`.
        5. The minimum time for all tasks to be executed is the maximum
           of these finish times across all processors.

        Args:
            processorTime: List of processor availability times.
            tasks: List of task execution times.

        Returns:
            The minimum time when all tasks are executed.
        """
        # Sort processor availability times (ascending)
        processorTime.sort()

        # Sort task execution times (ascending)
        tasks.sort()

        n = len(processorTime)
        total_tasks = len(tasks) # which is 4 * n

        max_finish_time = 0

        # Iterate through processors
        for i in range(n):
            # The i-th earliest processor (processorTime[i]) is assigned
            # the i-th block of 4 longest tasks.
            # The indices for the tasks assigned to processor i (0-indexed)
            # start from the end of the sorted tasks list.
            # For i=0, tasks are tasks[4n-4]..tasks[4n-1]. Max is tasks[4n-1]. Index 4n - 1 - 4*0.
            # For i=1, tasks are tasks[4n-8]..tasks[4n-5]. Max is tasks[4n-5]. Index 4n - 1 - 4*1.
            # For processor i, the tasks are tasks[4n - 4*(i+1)]..tasks[4n - 4*i - 1].
            # The maximum task time for processor i is tasks[4*n - 1 - 4*i].
            # Note: task index goes from 4n-1 down to 0 in steps of 4 for the max task in each group.
            # For processor i (0-indexed), the index of the maximum task it receives is
            # total_tasks - 1 - i * 4.
            max_task_index_for_processor_i = total_tasks - 1 - i * 4
            
            finish_time = processorTime[i] + tasks[max_task_index_for_processor_i]

            max_finish_time = max(max_finish_time, finish_time)

        return max_finish_time