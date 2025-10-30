class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        
        # Compute prefix sums for O(1) subarray sum calculations
        prefix_sum = [0]
        for num in nums:
            prefix_sum.append(prefix_sum[-1] + num)
        
        # Calculate the sum for each subarray
        for i in range(n):
            start = max(0, i - nums[i])
            # Use prefix_sum to get the subarray sum in O(1) time
            total_sum += prefix_sum[i+1] - prefix_sum[start]
        
        return total_sum