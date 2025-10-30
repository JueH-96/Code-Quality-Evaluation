class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort the processorTime and tasks arrays
        processorTime.sort()
        tasks.sort(reverse=True)

        # Initialize the minimum time to the maximum possible value
        min_time = 0

        # Iterate through the processors and assign tasks
        for i in range(len(processorTime)):
            # Calculate the time taken by the current processor to execute its tasks
            current_processor_time = processorTime[i] + max(tasks[i * 4], tasks[i * 4 + 1], tasks[i * 4 + 2], tasks[i * 4 + 3])
            # Update the minimum time if the current processor's time is greater
            min_time = max(min_time, current_processor_time)

        return min_time