from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = nums.length
        if n == 0:
            return 0

        # Initialize max_cost and min_cost arrays
        max_cost = [0] * n
        min_cost = [0] * n

        # Base case
        max_cost[0] = nums[0]
        min_cost[0] = nums[0]

        # Fill the max_cost and min_cost arrays
        for i in range(1, n):
            max_cost[i] = max(nums[i], max_cost[i-1] + nums[i] * (-1 if i % 2 == 0 else 1))
            min_cost[i] = min(nums[i], min_cost[i-1] + nums[i] * (-1 if i % 2 == 0 else 1))

        # The maximum total cost will be the maximum value in max_cost array
        return max(max_cost)

# Example usage:
solution = Solution()
print(solution.maximumTotalCost([1, -2, 3, 4]))  # Output: 10
print(solution.maximumTotalCost([1, -1, 1, -1]))  # Output: 4
print(solution.maximumTotalCost([0]))            # Output: 0
print(solution.maximumTotalCost([1, -1]))         # Output: 2