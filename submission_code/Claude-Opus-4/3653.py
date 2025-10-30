class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Calculate prefix sums for efficient subarray sum calculation
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        
        max_sum = float('-inf')
        
        # Try all possible lengths that are multiples of k
        for length in range(k, n + 1, k):
            # For current length, find maximum sum subarray
            for i in range(n - length + 1):
                # Sum of subarray from index i to i+length-1
                current_sum = prefix[i + length] - prefix[i]
                max_sum = max(max_sum, current_sum)
        
        return max_sum