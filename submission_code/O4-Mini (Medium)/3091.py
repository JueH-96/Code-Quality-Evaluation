from typing import List
from collections import Counter

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        # Count occurrences of each number
        freq = Counter(nums)
        # Handle zeros separately: they contribute multiplicatively
        zero_count = freq.pop(0, 0)
        
        # Prepare dp array: dp[s] = number of ways to make sum s (without zeros)
        dp = [0] * (r + 1)
        dp[0] = 1
        
        # Process each distinct non-zero value with its count
        for v, cnt in freq.items():
            # Copy old dp
            old = dp
            dp = [0] * (r + 1)
            # We do bounded knapsack via sliding window on residues mod v
            for rem in range(v):
                # Build the sequence of old sums for this residue class
                seq = []
                # s = j*v + rem  => j = 0,1,... such that s <= r
                max_j = (r - rem) // v
                for j in range(max_j + 1):
                    seq.append(old[j * v + rem])
                
                # Sliding window sum of width cnt+1 over seq
                window_sum = 0
                left = 0
                for j in range(len(seq)):
                    window_sum = (window_sum + seq[j]) % MOD
                    # If window too large, remove the leftmost
                    if j - left + 1 > cnt + 1:
                        window_sum = (window_sum - seq[left]) % MOD
                        left += 1
                    # window_sum now is sum_{k=0..cnt} seq[j-k]
                    dp[j * v + rem] = window_sum
                
            # dp now updated for this (v,cnt)
        
        # Factor in zeros: for any sum s, we can choose 0..zero_count zeros
        if zero_count > 0:
            mult = zero_count + 1
            for s in range(r + 1):
                dp[s] = dp[s] * mult % MOD
        
        # Sum up the ways for sums in [l..r]
        ans = sum(dp[l:r+1]) % MOD
        return ans