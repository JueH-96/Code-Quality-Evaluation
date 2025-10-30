from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort tasks descending by duration, and processors ascending by availability.
        tasks.sort(reverse=True)
        processorTime.sort()
        
        # Assign the i-th largest task to the floor(i/4)-th earliest processor.
        # Each processor has exactly 4 cores, i.e. 4 tasks.
        ans = 0
        for i, t in enumerate(tasks):
            finish = processorTime[i // 4] + t
            if finish > ans:
                ans = finish
        
        return ans