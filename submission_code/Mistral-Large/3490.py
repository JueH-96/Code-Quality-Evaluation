from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Initialize variables to keep track of the longest valid subsequences
        # ending with an even sum and an odd sum
        max_even = 0
        max_odd = 0

        # Initialize variables to keep track of the current valid subsequences
        # ending with an even sum and an odd sum
        current_even = 0
        current_odd = 0

        for i in range(1, len(nums)):
            sum_pair = nums[i] + nums[i - 1]
            if sum_pair % 2 == 0:
                # If the sum is even, update the current even subsequence
                current_even = max(current_even, current_odd) + 1
                current_odd = 0
            else:
                # If the sum is odd, update the current odd subsequence
                current_odd = max(current_odd, current_even) + 1
                current_even = 0

            # Update the maximum lengths
            max_even = max(max_even, current_even)
            max_odd = max(max_odd, current_odd)

        # The longest valid subsequence is the maximum of the two
        return max(max_even, max_odd) + 1

# Example usage:
# solution = Solution()
# print(solution.maximumLength([1, 2, 3, 4]))  # Output: 4
# print(solution.maximumLength([1, 2, 1, 1, 2, 1, 2]))  # Output: 6
# print(solution.maximumLength([1, 3]))  # Output: 2