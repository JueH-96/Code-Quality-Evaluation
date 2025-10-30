class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        if x == 0:
            return 0

        # Initialize the minimum absolute difference to a large value
        min_diff = float('inf')

        # Iterate through the array
        for i in range(n):
            for j in range(i + x, n):
                min_diff = min(min_diff, abs(nums[i] - nums[j]))

        return min_diff