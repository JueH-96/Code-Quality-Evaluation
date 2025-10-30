from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # Sort the processorTime and tasks in ascending order
        processorTime.sort()
        tasks.sort()

        # Initialize the result variable to store the maximum time taken
        result = 0

        # Loop through each processor
        for i in range(len(processorTime)):
            # Calculate the time taken for the current processor to execute its tasks
            time_taken = max(
                processorTime[i] + tasks[i * 4],
                processorTime[i] + tasks[i * 4 + 1],
                processorTime[i] + tasks[i * 4 + 2],
                processorTime[i] + tasks[i * 4 + 3]
            )
            # Update the result with the maximum time taken so far
            result = max(result, time_taken)

        return result

# Example usage:
# sol = Solution()
# print(sol.minProcessingTime([8,10], [2,2,3,1,8,7,4,5]))  # Output: 16
# print(sol.minProcessingTime([10,20], [2,3,1,2,5,8,4,3]))  # Output: 23