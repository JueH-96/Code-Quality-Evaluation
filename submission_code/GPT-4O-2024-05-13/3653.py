class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = float('-inf')
        
        # Iterate over all possible subarray lengths that are divisible by k
        for length in range(k, n + 1, k):
            # Calculate the sum of the first subarray of this length
            current_sum = sum(nums[:length])
            max_sum = max(max_sum, current_sum)
            
            # Use sliding window to calculate the sum of other subarrays of this length
            for i in range(length, n):
                current_sum += nums[i] - nums[i - length]
                max_sum = max(max_sum, current_sum)
        
        return max_sum