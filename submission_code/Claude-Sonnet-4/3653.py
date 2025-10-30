class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Calculate prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        max_sum = float('-inf')
        
        # For each possible length that's divisible by k
        for length in range(k, n + 1, k):
            # Check all subarrays of this length
            for start in range(n - length + 1):
                end = start + length
                current_sum = prefix[end] - prefix[start]
                max_sum = max(max_sum, current_sum)
        
        return max_sum