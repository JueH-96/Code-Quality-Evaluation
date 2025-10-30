from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        operations = 0
        while len(nums) > 0:
            # Remove the first 3 elements or all remaining elements if fewer than 3
            nums = nums[3:] if len(nums) > 3 else []
            operations += 1
            # Check if the remaining elements are distinct
            if len(nums) == len(set(nums)):
                break
        return operations

# Example usage:
# solution = Solution()
# print(solution.minimumOperations([1,2,3,4,2,3,3,5,7]))  # Output: 2
# print(solution.minimumOperations([4,5,6,4,4]))  # Output: 2
# print(solution.minimumOperations([6,7,8,9]))  # Output: 0