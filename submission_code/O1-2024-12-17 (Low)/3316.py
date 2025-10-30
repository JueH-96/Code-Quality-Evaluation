class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        import sys
        sys.setrecursionlimit(10**7)
        from bisect import bisect_left
        
        MOD = 10**9 + 7
        n = len(nums)
        
        # Sort the array
        nums.sort()
        
        # Collect all possible (nonnegative) differences
        diffs = set()
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(nums[j] - nums[i])
        # Convert to sorted list
        diffs = sorted(diffs)
        
        # Edge case: if no differences found (should not happen for n>=2) 
        # or if diffs is empty, but realistically we might just rely on the loop.
        if not diffs:
            # All elements are the same, minDiff = 0 for any subsequence (of length >=2)
            # The sum of minDiff over all subsequences is 0.
            return 0
        
        # We will append one more difference larger than any possible difference
        # so that we can compute # subsets whose min diff >= next_diff easily.
        max_diff = diffs[-1]  # largest difference
        diffs_plus = diffs + [max_diff + 1]
        
        # Precompute a fast nCr or do DP.  Since n<=50, we can do direct DP if needed.
        # But here we need a different DP: count how many subsequences of length k 
        # have pairwise difference >= d.
        
        # Memo for count_subsets(d)
        from functools import lru_cache
        
        # We'll build a function that, given a gap d, returns the number of 
        # size-k subsets with all pairwise differences >= d.
        def count_subsets(d):
            # Precompute for each index i, the smallest j>i such that nums[j] - nums[i] >= d
            # If none, we set next_idx[i] = n (meaning we can't pick another from here).
            next_idx = [n]*n
            j = 0
            for i in range(n):
                while j < n and nums[j] - nums[i] < d:
                    j += 1
                next_idx[i] = j if j < n else n
            
            # dp[i][x] = number of ways to pick a subset of size x from nums[i..],
            # with the pairwise-diff constraint.
            dp = [[0]*(k+1) for _ in range(n+1)]
            # Base case: dp[anything][0] = 1 (one way to choose an empty subset)
            for i in range(n+1):
                dp[i][0] = 1
            
            # Fill dp from the back
            for i in range(n-1, -1, -1):
                for x in range(1, k+1):
                    # Skip nums[i]
                    ways = dp[i+1][x]
                    # Pick nums[i], then next chosen index must be next_idx[i]
                    ni = next_idx[i]
                    if x-1 >= 0 and ni <= n:
                        ways += dp[ni][x-1]
                    dp[i][x] = ways % MOD
            
            return dp[0][k]
        
        # We want: sum_{S} minDiff(S).
        # Group subsets by their minDiff. 
        # Let F(d) = # of subsets of size k with pairwise difference >= d.
        # Then # with minDiff = d_i = F(d_i) - F(d_{i+1}).
        # Sum = Σ d_i * (F(d_i) - F(d_{i+1})).
        
        # Precompute F(d) for each d in diffs_plus
        # Store in an array Fvals
        Fvals = []
        memo = {}
        for d in diffs_plus:
            c = count_subsets(d)
            Fvals.append(c)
        
        # Now compute the sum
        ans = 0
        for i in range(len(diffs)):
            diff_here = diffs[i]
            count_here = (Fvals[i] - Fvals[i+1]) % MOD
            ans = (ans + diff_here * count_here) % MOD
        
        return ans % MOD