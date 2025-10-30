class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_sum = 0
        
        # Generate all subarrays with length from 1 to k
        for length in range(1, k + 1):
            for start in range(n - length + 1):
                # Get the subarray
                subarray = nums[start:start + length]
                
                # Find min and max of this subarray
                min_val = min(subarray)
                max_val = max(subarray)
                
                # Add their sum to total
                total_sum += min_val + max_val
        
        return total_sum