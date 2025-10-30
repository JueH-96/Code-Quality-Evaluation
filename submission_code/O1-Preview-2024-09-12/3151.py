class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        processorTime.sort()
        tasks.sort(reverse=True)
        n = len(processorTime)
        max_time = 0
        for i in range(n):
            processor_available_time = processorTime[i]
            assigned_tasks = tasks[4*i:4*(i+1)]
            finishing_times = [processor_available_time + task_time for task_time in assigned_tasks]
            processor_finishing_time = max(finishing_times)
            max_time = max(max_time, processor_finishing_time)
        return max_time