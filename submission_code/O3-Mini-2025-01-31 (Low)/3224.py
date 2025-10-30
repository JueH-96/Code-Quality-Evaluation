from typing import List

MOD = 10**9 + 7

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        # Precompute factorials and inverse factorials up to n (n can be as high as 10^5)
        max_val = n
        fact = [1] * (max_val + 1)
        inv_fact = [1] * (max_val + 1)
        for i in range(2, max_val + 1):
            fact[i] = fact[i-1] * i % MOD
        # Fermat inverse using pow with mod MOD (MOD is prime)
        inv_fact[max_val] = pow(fact[max_val], MOD - 2, MOD)
        for i in range(max_val, 0, -1):
            inv_fact[i-1] = inv_fact[i] * i % MOD

        # Determine segments of non-infected children
        # Left segment: from position 0 to sick[0]-1, if any.
        left = sick[0]  # number of children before the first sick.
        # Right segment: from sick[-1]+1 to n-1.
        right = n - 1 - sick[-1]
        
        # In-between gaps: for each consecutive pair of sick children.
        gaps = []
        for i in range(len(sick)-1):
            gap_len = sick[i+1] - sick[i] - 1
            if gap_len > 0:
                gaps.append(gap_len)
                
        # Total number of non-infected children
        m = left + right + sum(gaps)
        
        # The overall interleaving among segments:
        # We will infect m children in some order.
        # But the order within each contiguous segment is forced in
        # one-infection-at-a-time scenario when there's only one available candidate.
        # However, for gap segments (which have infected boundaries from both sides),
        # the infection can progress from both ends. In a gap of length L (>=1),
        # the number of ways to infect that segment (keeping the order enforced from each end)
        # is the number of interleavings of the left-chain and right-chain.
        # In any gap, initially two neighbors (one from left boundary and one from right boundary) are infected.
        # Then, if we always maintain the order from each end,
        # the relative order in the left chain (which must be the increasing order from the left)
        # and in the right chain (which is the decreasing order from the right)
        # is fixed.
        # It turns out that the number of valid sequences to infect the gap is:
        #    f(L) = 2^(L-1)   for L >= 1,
        # with f(0)=1 by convention.
        # (Intuition: after the first infection, at each subsequent infection you have two choices)
        
        gap_sequences = 1
        for L in gaps:
            if L >= 1:
                gap_sequences = gap_sequences * pow(2, L - 1, MOD) % MOD
        
        # Now, aside from the forced orderings inside segments,
        # different segments can be interleaved arbitrarily.
        # Suppose we have segments of sizes: left, right, and for each gap: gap_i.
        # The number of ways to interleave these orders is:
        #    total = m! / ( left! * right! * (gap_1)! * (gap_2)! * ... ).
        #
        # Multiply that by the sequences count within gap segments.
        interleave = fact[m]
        interleave = interleave * inv_fact[left] % MOD
        interleave = interleave * inv_fact[right] % MOD
        for L in gaps:
            interleave = interleave * inv_fact[L] % MOD
        
        # Final answer
        ans = interleave * gap_sequences % MOD
        return ans