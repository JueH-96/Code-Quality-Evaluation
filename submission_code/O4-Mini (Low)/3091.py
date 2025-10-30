from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        from collections import Counter
        
        cnt = Counter(nums)
        # Handle zeros separately: including 0 can be chosen 0..cnt0 times, multiplies ways
        zero_count = cnt.pop(0, 0)
        zero_mult = pow(zero_count + 1, 1, MOD)  # (cnt0+1) ways, mod
        
        # dp[s] = number of ways to reach sum s with processed values
        dp = [0] * (r + 1)
        dp[0] = 1
        
        # Process each distinct non-zero value with its count
        for v, c in cnt.items():
            w = v
            old = dp
            new = [0] * (r + 1)
            # For each residue class mod w
            for rem in range(w):
                # Build the sequence of old_dp at positions rem + t*w
                seq = []
                idxs = []
                j = rem
                while j <= r:
                    seq.append(old[j])
                    idxs.append(j)
                    j += w
                L = len(seq)
                if L == 0:
                    continue
                # Build prefix sums of seq
                prefix = [0] * L
                prefix[0] = seq[0]
                for i in range(1, L):
                    prefix[i] = (prefix[i-1] + seq[i]) % MOD
                # For each position t in sequence compute sum of last c+1 values
                for t in range(L):
                    # sum over seq[max(0, t-c) .. t]
                    if t <= c:
                        val = prefix[t]
                    else:
                        val = (prefix[t] - prefix[t - c - 1]) % MOD
                    new[idxs[t]] = val
            dp = new
        
        # Sum dp[s] for s in [l..r], multiply by zero_mult, mod
        ans = sum(dp[l:r+1]) % MOD
        ans = (ans * zero_mult) % MOD
        return ans