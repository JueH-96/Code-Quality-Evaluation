class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        processorTime.sort()
        tasks.sort(reverse=True)

        n = len(processorTime)
        max_finish_time = 0

        for i in range(n):
            start_time = processorTime[i]
            # The 4 tasks assigned to processor i will be the tasks at indices 4*i, 4*i+1, 4*i+2, 4*i+3 in the sorted tasks list.
            # Since tasks are sorted in descending order, the longest task among these is at index 4*i.
            longest_task_time = tasks[i * 4]
            finish_time = start_time + longest_task_time
            max_finish_time = max(max_finish_time, finish_time)

        return max_finish_time