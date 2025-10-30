from bisect import bisect_left
from typing import List

MOD = 10**9 + 7

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        # sort the array along with their values
        nums.sort()
        n = len(nums)
        
        # Precompute candidate differences.
        # Only differences that appear (or plus one) between any two numbers are needed.
        cand = set()
        for i in range(n):
            for j in range(i+1, n):
                d = nums[j] - nums[i]
                cand.add(d)
        # We'll also want to count f(d) for d+1. So add d+1 for each candidate while keeping nonnegative.
        cand2 = set()
        for d in cand:
            if d+1 <= 2 * (10**8):
                cand2.add(d+1)
        cands = sorted(cand.union(cand2))
        
        # We'll memoize DP results for a given d value.
        # f(d): count number of k-length subsequences whose adjacent differences are all >= d.
        dp_cache = {}
        
        def count_subseq(d: int) -> int:
            # returns f(d) modulo MOD.
            if d in dp_cache:
                return dp_cache[d]
            # Precompute for every index the first index j with nums[j] >= nums[i] + d.
            nextIdx = [None] * n
            for i in range(n):
                # We want smallest j > i with nums[j] >= nums[i] + d.
                pos = bisect_left(nums, nums[i] + d, i+1, n)
                nextIdx[i] = pos  # note: pos can be n (meaning no valid index)
            
            # dp[length][i] will be the count of subsequences of length "length" ending at index i.
            # We need subsequences of length k.
            # Initialize: subsequences of length 1: each index can be chosen.
            dp = [[0]*n for _ in range(k+1)]
            for i in range(n):
                dp[1][i] = 1
            
            # For lengths from 1 to k-1, extend the subsequence.
            for length in range(1, k):
                # For each ending index i for the current subsequence:
                # We try to extend with an index j >= nextIdx[i] (so that nums[j]-nums[i]>=d).
                # For every i, if nextIdx[i] = pos, then for j from pos to n-1 we add.
                for i in range(n):
                    if dp[length][i]:
                        pos = nextIdx[i]
                        if pos < n:
                            # add dp[length][i] to every dp[length+1][j] with j from pos to n-1,
                            # but we can try to avoid an inner loop by a cumulative sum.
                            # However, since n is small (<=50) we can do a direct loop.
                            add_val = dp[length][i]
                            for j in range(pos, n):
                                dp[length+1][j] = (dp[length+1][j] + add_val) % MOD
            
            total = sum(dp[k]) % MOD
            dp_cache[d] = total
            return total
        
        # Now, for each candidate difference, compute f(d), f(d+1) then add d*(f(d) - f(d+1)).
        ans = 0
        for d in cands:
            count_d = count_subseq(d)
            count_d_plus = count_subseq(d+1)
            cnt = (count_d - count_d_plus) % MOD
            ans = (ans + d * cnt) % MOD
            
        return ans