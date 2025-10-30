import bisect

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        if not nums:
            return 0
        
        max_val = max(nums)
        n = len(nums)
        binary = [1 if x == max_val else 0 for x in nums]
        
        # Compute the prefix sum array
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + binary[i - 1]
        
        total = 0
        
        for r in range(n):
            current_sum = prefix_sum[r + 1]
            target = current_sum - k
            
            # Search in the prefix_sum array up to index r+1 (inclusive)
            hi = r + 2  # since bisect is exclusive on the upper bound
            if hi > len(prefix_sum):
                hi = len(prefix_sum)
            
            # Find the rightmost index where prefix_sum[index] <= target
            idx = bisect.bisect_right(prefix_sum, target, 0, hi) - 1
            
            if idx >= 0:
                total += (idx + 1)  # number of valid starting points
        
        return total