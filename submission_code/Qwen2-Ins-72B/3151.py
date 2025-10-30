class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        tasks.sort(reverse=True)
        processorTime.sort()
        res = 0
        for i in range(len(tasks)):
            res = max(res,processorTime[i//4]+tasks[i])
        return res