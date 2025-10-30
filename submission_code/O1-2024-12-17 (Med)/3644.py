class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        
        # Calculate prefix sums
        prefix_sums = [0] * (n+1)
        for i in range(n):
            prefix_sums[i+1] = prefix_sums[i] + nums[i]
        
        # Variable to track the minimum subarray sum > 0
        min_sum = float('inf')
        
        # Check subarrays of length from l to r
        for length in range(l, r+1):
            for start in range(n - length + 1):
                current_sum = prefix_sums[start + length] - prefix_sums[start]
                if current_sum > 0:
                    min_sum = min(min_sum, current_sum)
        
        # If we didn't find any subarray with sum > 0
        return min_sum if min_sum != float('inf') else -1