class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        import math
        
        n = len(nums)
        # Prefix sums f[i] = sum of nums[:i]
        # so the sum of nums[l..r] = f[r+1] - f[l]
        f = [0] * (n+1)
        for i in range(n):
            f[i+1] = f[i] + nums[i]
        
        # We define "best[i][j]" as the maximum strength using j subarrays
        # from the first i elements of nums (nums[0..i-1]), but we will only
        # store one dimension for j-1 (prev) and one for j (cur) to save memory.
        
        # best array for j=0: picking 0 subarrays => strength = 0
        # for i>0, best[i][0] = 0; for j>0, best[0][j] = -inf
        
        NEG_INF = float('-inf')
        
        prev = [0] + [NEG_INF]*n  # best[..][0] is 0 for all i, because 0 subarrays => strength=0
                                 # (We keep length n+1, index i from 0..n.)
        
        # c_j = (k-j+1)* (+1 or -1), sign alternates: + if j is odd, - if j is even.
        
        for j in range(1, k+1):
            sign_j = 1 if (j % 2) == 1 else -1
            c_j = (k - j + 1) * sign_j
            
            cur = [NEG_INF] * (n+1)
            # best[0][j]: no elements chosen but we want j subarrays => not possible if j>0 => -inf
            cur[0] = NEG_INF
            
            # We maintain a running maximum M = max_{x in [0..i]} [ prev[x] - c_j*f[x] ]
            # so that dp[i] = c_j*f[i] + M.
            M = NEG_INF
            
            for i in range(n+1):
                # Update M using the old best[x] before i, i.e. best[i] from j-1
                if prev[i] != NEG_INF:
                    # prev[i] corresponds to best[i][j-1]
                    candidate = prev[i] - c_j*f[i]
                    if candidate > M:
                        M = candidate
                
                if i > 0:
                    # dp[i][j] = c_j*f(i) + M
                    # i.e. subarray ends at index i-1
                    if M != NEG_INF:
                        dp_val = c_j * f[i] + M
                        if dp_val > cur[i]:
                            cur[i] = dp_val
                    
                    # best[i][j] = max(best[i-1][j], dp[i][j])
                    if cur[i-1] != NEG_INF and cur[i-1] > cur[i]:
                        cur[i] = cur[i-1]
            
            prev = cur
        
        # The answer is best[n][k], stored in prev[n].
        return prev[n]