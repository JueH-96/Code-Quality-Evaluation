class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        processorTime.sort()
        tasks.sort(reverse=True)
        
        result = 0
        for i in range(len(processorTime)):
            current_max = processorTime[i] + tasks[4 * i]
            if current_max > result:
                result = current_max
        return result