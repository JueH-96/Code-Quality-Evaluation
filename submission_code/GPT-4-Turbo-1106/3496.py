from typing import List

class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        workerTimes.sort()
        time_taken = []
        for i, time in enumerate(workerTimes):
            # Calculate the time taken by each worker to reduce the mountain height
            # by the number of units equal to their index + 1 (since workers are 0-indexed)
            units = i + 1
            worker_time = sum(time * j for j in range(1, units + 1))
            time_taken.append(worker_time)
            mountainHeight -= units
            if mountainHeight <= 0:
                break
        # If the mountain height is still greater than 0 after all workers have done their part,
        # we need to distribute the remaining work among the workers
        while mountainHeight > 0:
            for i in range(len(workerTimes)):
                if mountainHeight <= 0:
                    break
                # Each worker can reduce one more unit of the mountain height
                time_taken[i] += workerTimes[i] * (i + 1)
                mountainHeight -= 1
        return max(time_taken)

# Example usage:
# sol = Solution()
# print(sol.minNumberOfSeconds(4, [2,1,1])) # Output: 3
# print(sol.minNumberOfSeconds(10, [3,2,2,4])) # Output: 12
# print(sol.minNumberOfSeconds(5, [1])) # Output: 15