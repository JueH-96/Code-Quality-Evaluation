class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        n = len(processorTime)
        processorTime.sort()
        tasks_sorted = sorted(tasks, reverse=True)
        max_time = 0
        for i in range(n):
            group_max = tasks_sorted[4 * i]
            finish = processorTime[i] + group_max
            if finish > max_time:
                max_time = finish
        return max_time