from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Build prefix sums: prefix[i] = sum of nums[0..i-1]
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        # dp_prev[j] = max strength selecting (i-1) subarrays in nums[0..j]
        # Initialize for i=0: zero strength for zero subarrays
        dp_prev = [0] * n
        
        # Iterate over number of subarrays selected i = 1..k
        for i in range(1, k+1):
            W = k - i + 1                            # weight magnitude
            S = 1 if (i % 2) == 1 else -1            # sign: + for odd i, - for even i
            dp_cur = [0] * n
            # best stores max over t<=j of (dp_prev[t-1] - S*W*prefix[t])
            # for t=0, dp_prev[-1] is 0, prefix[0]=0
            best = 0  # = 0 - S*W*0
            
            for j in range(n):
                # candidate if we end the i-th subarray at j
                # term = S*W*(prefix[j+1]) + best
                cur = S * W * prefix[j+1] + best
                # carry over the best of not ending at j vs ending at j
                if j == 0:
                    dp_cur[j] = cur
                else:
                    dp_cur[j] = max(dp_cur[j-1], cur)
                
                # now update best to include t = j+1 for future j's:
                # best = max(best, dp_prev[j] - S*W*prefix[j+1])
                v = dp_prev[j] - S * W * prefix[j+1]
                if v > best:
                    best = v
            
            dp_prev = dp_cur
        
        # answer = dp_prev[n-1] for selecting k subarrays in full array
        return dp_prev[n-1]