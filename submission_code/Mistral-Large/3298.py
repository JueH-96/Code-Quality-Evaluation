from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        # Sort the array
        nums.sort()

        max_length = 0
        current_length = 1
        increment_used = False

        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                current_length += 1
            elif nums[i] == nums[i - 1] + 1:
                current_length += 1
            elif nums[i] == nums[i - 1] + 2 and not increment_used:
                current_length += 1
                increment_used = True
            else:
                max_length = max(max_length, current_length)
                current_length = 1
                increment_used = False

        max_length = max(max_length, current_length)
        return max_length

# Example usage:
# solution = Solution()
# print(solution.maxSelectedElements([2,1,5,1,1]))  # Output: 3
# print(solution.maxSelectedElements([1,4,7,10]))   # Output: 1