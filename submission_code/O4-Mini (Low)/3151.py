from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Number of processors
        n = len(processorTime)
        # Sort processor availability times ascending
        processorTime.sort()
        # Sort tasks descending
        tasks.sort(reverse=True)
        # Assign the i-th earliest processor the i-th block of 4 largest remaining tasks
        # The block for processor i is tasks[4*i : 4*i+4], whose max is tasks[4*i]
        # Compute the makespan per processor and take the overall maximum
        ans = 0
        for i in range(n):
            # finishing time = availability + max task duration in its 4 tasks
            finish = processorTime[i] + tasks[4*i]
            if finish > ans:
                ans = finish
        return ans