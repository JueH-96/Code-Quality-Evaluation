from typing import List

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Compute factorials and inverse factorials up to n
        max_n = n
        fact = [1] * (max_n + 1)
        invfact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fact[i] = fact[i-1] * i % MOD
        # Fermat's little theorem for inverse factorials
        invfact[max_n] = pow(fact[max_n], MOD - 2, MOD)
        for i in range(max_n, 0, -1):
            invfact[i-1] = invfact[i] * i % MOD
        
        # Build segments of healthy children
        segments = []
        # Leading segment (before first sick)
        first_seg = sick[0] - 0
        if first_seg > 0:
            segments.append(first_seg)
        # Internal segments
        for i in range(len(sick) - 1):
            gap = sick[i+1] - sick[i] - 1
            if gap > 0:
                segments.append(gap)
        # Trailing segment (after last sick)
        last_seg = (n - 1) - sick[-1]
        if last_seg > 0:
            segments.append(last_seg)
        
        # Total number of healthy children
        total_healthy = sum(segments)
        
        # Multinomial coefficient: total_healthy! / prod(seg_len!)
        result = fact[total_healthy]
        for seg in segments:
            result = result * invfact[seg] % MOD
        
        # For each internal segment (not the first or last),
        # multiply by 2^(seg_len - 1)
        # Identify internal segments by checking
        # we skip the first and last if they touch the ends.
        # Reconstruct precisely:
        internal_segments = []
        # between sick positions only
        for i in range(len(sick) - 1):
            gap = sick[i+1] - sick[i] - 1
            if gap > 0:
                internal_segments.append(gap)
        
        for seg in internal_segments:
            # each internal segment of length k contributes 2^(k-1)
            if seg > 1:
                result = result * pow(2, seg - 1, MOD) % MOD
        
        return result