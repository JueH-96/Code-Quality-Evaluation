from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        # Calculate the difference array
        diff = [target[i] - nums[i] for i in range(len(nums))]

        # Calculate the total number of operations needed
        total_operations = sum(abs(d) for d in diff)

        # Since each operation can increment or decrement by 1,
        # we divide the total by 2 and round up to get the minimum number of operations
        return (total_operations + 1) // 2

# Example usage:
# sol = Solution()
# print(sol.minimumOperations([3,5,1,2], [4,6,2,4]))  # Output: 2
# print(sol.minimumOperations([1,3,2], [2,1,4]))      # Output: 5