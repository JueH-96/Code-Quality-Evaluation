class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        import sys
        sys.setrecursionlimit(10**7)
        
        MOD = 10**9 + 7
        
        # Precompute factorials and inverse factorials up to n (for combinations).
        # We will need up to n (the maximum number of non-infected children cannot exceed n).
        fact = [1]*(n+1)
        for i in range(1,n+1):
            fact[i] = fact[i-1]*i % MOD
        invfact = [1]*(n+1)
        invfact[n] = pow(fact[n], MOD-2, MOD)  # Fermat's little theorem for inverse
        for i in reversed(range(n)):
            invfact[i] = invfact[i+1]*(i+1) % MOD
        
        def comb(a, b):
            # Compute C(a,b) under MOD
            if b<0 or b> a:
                return 0
            return (fact[a]*invfact[b]%MOD)*invfact[a-b]%MOD
        
        # sick is sorted in increasing order per the problem statement.
        # Identify contiguous blocks of uninfected children.
        # A block is defined by [start, end] (inclusive) of positions that are not infected.
        # We'll collect lengths of these blocks, plus note whether each block is an "internal" block
        # (i.e. flanked on both sides by infected children) or an "edge" block (only one infected side).
        
        # Blocks can be:
        #   Left edge block: 0..(sick[0] - 1) if sick[0] > 0
        #   Middle blocks: (sick[i-1]+1)..(sick[i]-1) for i in [1..len(sick)-1]
        #   Right edge block: (sick[-1]+1)..(n-1) if sick[-1] < n-1
        
        blocks = []  # will hold tuples (L, is_internal)
        
        # 1) Left block
        if sick[0] > 0:
            L = sick[0]  # positions from 0 .. sick[0]-1
            blocks.append((L, False))  # edge block
        
        # 2) Middle blocks
        for i in range(1, len(sick)):
            start = sick[i-1] + 1
            end = sick[i] - 1
            L = end - start + 1
            if L > 0:
                blocks.append((L, True))  # internal block
        
        # 3) Right block
        if sick[-1] < n-1:
            L = (n-1) - sick[-1]
            blocks.append((L, False))  # edge block
        
        # Compute total number of uninfected children (sum of block lengths).
        total_uninfected = sum(L for (L, _) in blocks)
        
        # For each block:
        #   if is_internal => number of ways to infect that block alone = 2^(L-1)
        #   if not internal => only 1 way (forced from one side).
        # Then we must interleave the infection of these blocks in the final global sequence.
        # The number of ways to interleave block i of length L_i with block j of length L_j, etc.
        # is the multinomial coefficient total_uninfected! / (L_1! * L_2! * ...).
        
        # Compute the product of ways for each block, and multiply by the number of ways to interleave.
        ways_blocks = 1
        pow2 = [1]*(n+1)  # precompute powers of 2 up to n
        for i in range(1, n+1):
            pow2[i] = (pow2[i-1]*2) % MOD
        
        lengths = []
        for (L, is_internal) in blocks:
            lengths.append(L)
            if L > 0 and is_internal:
                # multiply by 2^(L-1)
                ways_blocks = (ways_blocks * pow2[L-1]) % MOD
        
        # Interleave factor = (total_uninfected)! / (L1! * L2! * ... * Lk!)
        #  = fact[total_uninfected] * Π invfact[L_i]
        interleave = fact[total_uninfected]
        for L in lengths:
            interleave = (interleave * invfact[L]) % MOD
        
        ans = (ways_blocks * interleave) % MOD
        return ans