from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        # Compute prefix sums: prefix[i] = sum(nums[0:i])
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        # A sufficiently small value to represent negative infinity.
        NEG_INF = -10**15
        
        # Base case: With 0 segments, the best sum you can have for any prefix is 0.
        prev_dp = [0] * (n + 1)
        
        # For each segment count from 1 to k, build the best scores for using that many segments.
        for seg in range(1, k + 1):
            # dp[i] will store the maximum sum when considering the first i elements with exactly seg segments.
            cur_dp = [NEG_INF] * (n + 1)
            
            # Precompute an auxiliary array "best" for the previous dp row.
            # best[i] = max_{x in [0, i]} (prev_dp[x] - prefix[x])
            # This will allow us to compute candidate = prefix[i] + max_{x in [0, i-m]} (prev_dp[x]-prefix[x])
            best = [NEG_INF] * (n + 1)
            best[0] = prev_dp[0] - prefix[0]
            for i in range(1, n + 1):
                val = prev_dp[i] - prefix[i]
                best[i] = best[i-1] if best[i-1] > val else val
            
            # Build cur_dp (dp for seg segments) using dp recurrence:
            # For i elements, either we skip the i-th element (carry cur_dp[i-1])
            # or we end a segment at index i-1. Ending a segment here means using a partition
            # where the segment length L is at least m. We can optimize this by writing:
            #   candidate = prefix[i] + best[i - m]
            # because if x is the index where previous segment ends, then 
            #   dp[i][seg] = dp[x][seg-1] + (prefix[i]-prefix[x]), and maximizing over x in [0, i-m]
            # is equivalent to prefix[i] + max_{x in [0, i-m]} (dp[x][seg-1]-prefix[x]).
            for i in range(1, n + 1):
                candidate = NEG_INF
                if i >= m:
                    candidate = prefix[i] + best[i - m]
                # Also, ensure the dp state is non-decreasing in i because we can always skip the current element.
                cur_dp[i] = cur_dp[i-1] if cur_dp[i-1] > candidate else candidate
            # Update for the next segment count.
            prev_dp = cur_dp
        
        return prev_dp[n]