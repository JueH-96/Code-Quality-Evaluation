class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        
        # Calculate prefix sum array
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i-1] + nums[i-1]
        
        min_sum = float('inf')
        found = False
        
        # Check all subarrays with length between l and r
        for i in range(n):
            for j in range(i + l - 1, min(i + r, n)):
                subarray_sum = prefix_sum[j + 1] - prefix_sum[i]
                if subarray_sum > 0:
                    min_sum = min(min_sum, subarray_sum)
                    found = True
        
        return min_sum if found else -1