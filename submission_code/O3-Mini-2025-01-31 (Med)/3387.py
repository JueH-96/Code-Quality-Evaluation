from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        mid = n // 2  # median index (for even, the larger of the two middles)

        ops = abs(nums[mid] - k)
        # For indices to the right of median, if they are below k, we need to increase them.
        for i in range(mid + 1, n):
            if nums[i] < k:
                ops += (k - nums[i])
        # For indices to the left of median, if they exceed k, we need to lower them.
        for i in range(0, mid):
            if nums[i] > k:
                ops += (nums[i] - k)
        return ops

# For testing purposes (not part of the solution interface):
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.minOperationsToMakeMedianK([2,5,6,8,5], 4))  # Output should be 2
    # Example 2:
    print(sol.minOperationsToMakeMedianK([2,5,6,8,5], 7))  # Output should be 3
    # Example 3:
    print(sol.minOperationsToMakeMedianK([1,2,3,4,5,6], 4))  # Output should be 0