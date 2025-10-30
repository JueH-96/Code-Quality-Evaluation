class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        # Calculate the difference array D where D[i] = target[i] - nums[i].
        D = [t - n for t, n in zip(target, nums)]
        n = len(nums)
        
        # If there's only one element, the required operations is just the absolute difference.
        if n == 1:
            return abs(D[0])
        
        # Compute the sum of absolute differences between consecutive elements in D.
        consecutive_diff_sum = 0
        for i in range(n - 1):
            consecutive_diff_sum += abs(D[i + 1] - D[i])
        
        # The result is given by (|D[0]| + sum_of_consecutive_diffs + |D[n-1]|) / 2
        # This formula counts the necessary subarray increments/decrements to match target.
        return (abs(D[0]) + consecutive_diff_sum + abs(D[n - 1])) // 2