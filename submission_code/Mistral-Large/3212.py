from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        last_occurrence = {}

        # Step 1: Find the last occurrence of each number
        for i, num in enumerate(nums):
            last_occurrence[num] = i

        # Step 2: Calculate the gaps between the last occurrences
        max_reach = -1
        partitions = 1
        current_gap = 0

        for i, num in enumerate(nums):
            max_reach = max(max_reach, last_occurrence[num])
            if max_reach == i:
                current_gap = i - max_reach
                partitions = (partitions * (current_gap + 1)) % MOD

        return partitions

# Example usage:
# sol = Solution()
# print(sol.numberOfGoodPartitions([1, 2, 3, 4]))  # Output: 8
# print(sol.numberOfGoodPartitions([1, 1, 1, 1]))  # Output: 1
# print(sol.numberOfGoodPartitions([1, 2, 1, 3]))  # Output: 2