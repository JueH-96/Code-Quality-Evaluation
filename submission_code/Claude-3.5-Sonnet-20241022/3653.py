class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Calculate prefix sums
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # Initialize result to smallest possible value
        max_sum = float('-inf')
        
        # For each possible length that's divisible by k
        for length in range(k, n + 1, k):
            # Slide window of size 'length' and find maximum sum
            curr_max = float('-inf')
            for start in range(n - length + 1):
                end = start + length
                curr_sum = prefix_sum[end] - prefix_sum[start]
                curr_max = max(curr_max, curr_sum)
            max_sum = max(max_sum, curr_max)
        
        return max_sum