class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        # We want to transform nums into target using subarray +/- 1 operations.
        # Let diff[i] = target[i] - nums[i].
        # A known result is that the minimum operations required is:
        #
        # ( |diff[0]| + ∑(i=1 to n-1) |diff[i] - diff[i-1]| + |diff[n-1]| ) / 2
        #
        # This formula correctly matches the examples given.

        n = len(nums)
        # Compute the difference array
        diff = [target[i] - nums[i] for i in range(n)]

        # Calculate the sum of absolute "changes" across the diff array
        total_changes = abs(diff[0])
        for i in range(1, n):
            total_changes += abs(diff[i] - diff[i-1])
        total_changes += abs(diff[n-1])

        # The result is half of this sum
        return total_changes // 2