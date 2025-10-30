from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort tasks in ascending order (smallest first).
        tasks_sorted = sorted(tasks)
        # Sort processors in descending order.
        processors_sorted = sorted(processorTime, reverse=True)
        n = len(processorTime)
        total_tasks = len(tasks_sorted)  # Should be equal to 4*n

        # Helper function: given candidate time T, check if we can assign tasks to all processors.
        def can_finish(T: int) -> bool:
            # We will assign tasks greedily.
            # Process the processor with the largest available time first (they have strictest constraints).
            task_index = 0
            for p in processors_sorted:
                # For processor with available time p, tasks in its group must have durations d such that:
                # p + d <= T, meaning d <= T - p.
                limit = T - p
                # There must be at least 4 tasks available.
                if task_index + 4 > total_tasks:
                    return False
                # Because tasks are sorted in ascending order, 
                # if the 4th smallest task in the available pool is <= limit, then all assigned tasks work.
                if tasks_sorted[task_index + 3] > limit:
                    return False
                task_index += 4
            return True

        # Determine search space.
        # Lower bound: the fastest time when a processor can finish a task.
        low = min(processorTime) + min(tasks)
        # Upper bound: worst-case finish time when a processor with maximum start time gets the longest task.
        high = max(processorTime) + max(tasks)
        
        result = high
        # Binary search for the minimum T that allows assignment.
        while low <= high:
            mid = (low + high) // 2
            if can_finish(mid):
                result = mid
                high = mid - 1
            else:
                low = mid + 1
        return result