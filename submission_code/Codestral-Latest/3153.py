class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7

        # Sort the array in descending order
        nums.sort(reverse=True)

        # Calculate the sum of squares of the first k elements
        max_sum = sum((nums[i] ** 2) % MOD for i in range(k)) % MOD

        return max_sum