from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return -1

        # Count the number of operations
        operations = 0

        # Iterate through the array
        for i in range(n):
            if nums[i] == 0:
                # If we can flip the next three elements
                if i + 2 < n:
                    # Flip the next three elements
                    nums[i] = 1 - nums[i]
                    nums[i + 1] = 1 - nums[i + 1]
                    nums[i + 2] = 1 - nums[i + 2]
                    operations += 1
                else:
                    # If we cannot flip the next three elements, return -1
                    return -1

        # Check if all elements are 1
        if all(x == 1 for x in nums):
            return operations
        else:
            return -1

# Example usage:
# sol = Solution()
# print(sol.minOperations([0,1,1,1,0,0]))  # Output: 3
# print(sol.minOperations([0,1,1,1]))      # Output: -1