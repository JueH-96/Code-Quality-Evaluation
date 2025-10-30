from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Fastest processors (smallest starting time) should handle the heaviest tasks
        processorTime.sort()               # ascending
        tasks.sort(reverse=True)           # descending
        
        # For every processor the finishing moment is:
        #   processorTime[i] + max duration among its 4 tasks.
        # After the sorts above, for processor i that maximum is tasks[4*i]
        return max(
            processorTime[i] + tasks[4 * i]
            for i in range(len(processorTime))
        )