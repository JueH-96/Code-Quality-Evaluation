from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)

        # Track the last time each index can be marked
        last_mark_time = [-1] * (n + 1)

        # Track the number of times each index appears in changeIndices
        index_count = [0] * (n + 1)
        for index in changeIndices:
            index_count[index] += 1

        # Track the remaining decrements needed for each index
        remaining_decrements = nums[:]

        for s in range(m):
            index = changeIndices[s]
            if remaining_decrements[index - 1] > 0:
                remaining_decrements[index - 1] -= 1
            if remaining_decrements[index - 1] == 0:
                last_mark_time[index] = s + 1

        # Check if all indices can be marked
        for i in range(1, n + 1):
            if last_mark_time[i] == -1 or index_count[i] == 0:
                return -1

        return max(last_mark_time[1:])

# Example usage:
# solution = Solution()
# print(solution.earliestSecondToMarkIndices([2,2,0], [2,2,2,2,3,2,2,1]))  # Output: 8
# print(solution.earliestSecondToMarkIndices([1,3], [1,1,1,2,1,1,1]))  # Output: 6
# print(solution.earliestSecondToMarkIndices([0,1], [2,2,2]))  # Output: -1