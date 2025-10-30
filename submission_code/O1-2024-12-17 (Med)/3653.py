class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        import math
        
        n = len(nums)
        prefix_sums = [0]*(n+1)
        
        # Build prefix sums array
        for i in range(n):
            prefix_sums[i+1] = prefix_sums[i] + nums[i]
        
        # best[r] will hold the minimum prefix sum encountered so far
        # for indices i with i mod k == r
        best = [math.inf]*k
        ans = -math.inf
        
        # Iterate over all prefix sums
        for i in range(n+1):
            r = i % k
            # If we have previously seen a prefix sum with the same mod r,
            # we can try to maximize the subarray sum by subtracting that prefix
            if best[r] != math.inf:
                ans = max(ans, prefix_sums[i] - best[r])
            
            # Update the minimum prefix sum for this mod class
            best[r] = min(best[r], prefix_sums[i])
        
        return ans