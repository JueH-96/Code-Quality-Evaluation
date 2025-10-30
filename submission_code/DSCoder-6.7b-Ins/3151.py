class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        n = len(processorTime)
        tasks_with_indexes = list(enumerate(tasks))
        tasks_with_indexes.sort(key=lambda x: x[1])  # sort tasks by their execution time
        processorTime.sort()  # sort processor times

        # assign tasks to processors based on their execution time
        assigned_tasks = [[] for _ in range(n)]
        for i in range(len(tasks)):
            assigned_tasks[i % n].append(tasks_with_indexes[i])

        # calculate the maximum execution time for each processor
        max_times = []
        for i in range(n):
            assigned_tasks[i].sort(key=lambda x: x[1])  # sort tasks by their execution time
            cur_time = processorTime[i]
            max_time = 0
            for j in range(len(assigned_tasks[i])):
                cur_time += assigned_tasks[i][j][1]
                max_time = max(max_time, cur_time)
            max_times.append(max_time)

        return min(max_times)  # return the minimum maximum execution time