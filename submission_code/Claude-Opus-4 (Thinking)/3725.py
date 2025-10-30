class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total = 0
        
        # Generate all subarrays with length from 1 to k
        for length in range(1, min(k, n) + 1):
            # For each possible starting position
            for start in range(n - length + 1):
                subarray = nums[start:start + length]
                total += min(subarray) + max(subarray)
        
        return total