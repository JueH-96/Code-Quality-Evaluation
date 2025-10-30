class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = float('-inf')
        
        # Compute prefix sums
        prefix_sums = [0] * (n + 1)
        for i in range(n):
            prefix_sums[i + 1] = prefix_sums[i] + nums[i]
        
        # Iterate over all possible subarray lengths that are divisible by k
        for length in range(k, n + 1, k):
            for start in range(n - length + 1):
                end = start + length
                subarray_sum = prefix_sums[end] - prefix_sums[start]
                max_sum = max(max_sum, subarray_sum)
        
        return max_sum