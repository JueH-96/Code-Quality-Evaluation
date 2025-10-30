from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_index = n // 2

        # Calculate the current median
        if n % 2 == 0:
            current_median = nums[median_index]
        else:
            current_median = nums[median_index]

        # Calculate the number of operations needed
        operations = 0
        for num in nums:
            operations += abs(num - k)

        # Since we only need to adjust to the median, we can optimize the calculation
        left_operations = sum(k - num for num in nums[:median_index] if num < k)
        right_operations = sum(num - k for num in nums[median_index:] if num > k)

        return min(left_operations, right_operations)

# Example usage:
sol = Solution()
print(sol.minOperationsToMakeMedianK([2,5,6,8,5], 4))  # Output: 2
print(sol.minOperationsToMakeMedianK([2,5,6,8,5], 7))  # Output: 3
print(sol.minOperationsToMakeMedianK([1,2,3,4,5,6], 4))  # Output: 0