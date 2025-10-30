class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort the processor times in ascending order
        processors_sorted = sorted(processorTime)
        # Sort the tasks in descending order
        tasks_sorted = sorted(tasks, reverse=True)
        max_time = 0
        n = len(processors_sorted)
        # Iterate through each processor and compute the maximum finish time
        for i in range(n):
            # The maximum task in the i-th group of 4 tasks
            current_max_task = tasks_sorted[4 * i]
            finish_time = processors_sorted[i] + current_max_task
            if finish_time > max_time:
                max_time = finish_time
        return max_time