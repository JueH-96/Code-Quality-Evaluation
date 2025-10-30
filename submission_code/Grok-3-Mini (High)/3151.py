class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        proc_sorted = sorted(processorTime)  # Sort processor times in ascending order
        tasks_sorted = sorted(tasks, reverse=True)  # Sort tasks in descending order
        return max(proc_sorted[i] + tasks_sorted[4 * i] for i in range(len(processorTime)))  # Compute max completion time