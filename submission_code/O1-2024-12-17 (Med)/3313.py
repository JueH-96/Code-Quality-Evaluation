class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        """
        We want to pick k disjoint subarrays to maximize the given 'strength':
            strength = sum[1]*k - sum[2]*(k-1) + sum[3]*(k-2) - ...
        which can be expressed as:
            strength = Σ_{i=1..k} [ ((-1)^(i+1)) * (k - i + 1) * (sum of i-th chosen subarray) ].

        The order of subarrays is left-to-right in the array (i.e. we cannot reorder them arbitrarily),
        so the i-th chosen subarray in left-to-right order is multiplied by a fixed factor
        factor[i] = ((-1)^(i+1)) * (k - i + 1).

        We use a dynamic programming approach akin to the "maximum sum of k non-overlapping subarrays",
        but each chosen subarray sum is weighted by factor[i], which may be positive or negative.
        
        Let prefixSum[i] = sum of nums[:i], with prefixSum[0] = 0.
        Define bestDP[i][j] = the maximum achievable strength using j subarrays within the first i elements.
        Also define dp[i][j] as the strength if the j-th subarray ends exactly at index i.

        Then:
            dp[i][j] = factor[j] * prefixSum[i] + max_{0 <= x < i} [ bestDP[x][j-1] - factor[j]*prefixSum[x] ]
        
        We iterate i from 1..n, and keep track of:
            M = max_{0 <= x < i} [ bestDP[x][j-1] - factor[j]*prefixSum[x] ]
        so that we can compute dp[i][j] in O(1).

        Finally, bestDP[i][j] = max(bestDP[i-1][j], dp[i][j]) so we can skip using index i or use it.

        We return bestDP[n][k] in the end.
        Since n*k <= 10^6, this O(n*k) solution is feasible.
        """

        import math

        n = len(nums)
        
        # Precompute prefix sums: prefixSum[i] = sum of nums[:i], 0-based
        prefixSum = [0] * (n+1)
        for i in range(n):
            prefixSum[i+1] = prefixSum[i] + nums[i]

        # Precompute the factors for each subarray index (1-based)
        # factor[j] = ((-1)^(j+1)) * (k - j + 1)
        # j runs from 1..k
        factors = [0] * (k+1)
        for j in range(1, k+1):
            sign = 1 if (j % 2) == 1 else -1
            factors[j] = sign * (k - j + 1)
        
        # bestDP[i][j] = max strength using j subarrays among first i elements
        # dp[i][j] = strength if the j-th subarray ends exactly at i (1-based "end index"),
        # but we'll keep i in 0..n referring to "prefixSum index".
        # We'll store them in 2D lists. Initialize with -inf for impossible.
        NEG_INF = float('-inf')
        dp = [[NEG_INF]*(k+1) for _ in range(n+1)]
        bestDP = [[NEG_INF]*(k+1) for _ in range(n+1)]
        
        # Base cases:
        # bestDP[0][0] = 0 (zero subarrays among zero elements => strength 0)
        bestDP[0][0] = 0
        for i in range(1, n+1):
            bestDP[i][0] = 0  # picking 0 subarrays always yields strength 0
        
        # Fill DP for each j in 1..k
        for j in range(1, k+1):
            factor_j = factors[j]
            # M will track max( bestDP[x][j-1] - factor_j * prefixSum[x] ) for x < i
            M = bestDP[0][j-1] - factor_j * prefixSum[0]  # x=0
            
            for i in range(1, n+1):
                # dp[i][j] = factor_j*prefixSum[i] + M
                dp[i][j] = factor_j * prefixSum[i] + M
                # bestDP[i][j] is the best we can do up to i using j subarrays
                bestDP[i][j] = max(bestDP[i-1][j], dp[i][j])
                
                # Now update M for the next i+1, considering x = i
                cand = bestDP[i][j-1] - factor_j * prefixSum[i]
                if cand > M:
                    M = cand

        return bestDP[n][k]