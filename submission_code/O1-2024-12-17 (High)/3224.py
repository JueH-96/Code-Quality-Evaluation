class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        import sys
        sys.setrecursionlimit(10**7)
        
        MOD = 10**9 + 7
        
        # Precompute factorials and inverse factorials up to n (<= 10^5).
        # This will allow fast computation of n! and combinations modulo MOD.
        fact = [1] * (n + 1)
        inv_fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i % MOD
        # Fermat's little theorem for inverse: inv_fact[n] = (fact[n])^(MOD-2) mod.
        inv_fact[n] = pow(fact[n], MOD - 2, MOD)
        for i in range(n - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        # Identify the "blocks" of consecutive uninfected children.
        # Blocks are separated by infected positions.
        # Each block can be one-sided (if it touches an end of the queue)
        # or two-sided (if it is strictly between two infected positions).
        
        # sick is sorted, so sick[0] is the leftmost infected, sick[-1] is rightmost.
        blocks = []  # will store tuples (length_of_block, type) where type=1 or 2
        
        k = len(sick)
        # Leftmost block (if the first infected is not at position 0)
        if sick[0] > 0:
            # length = sick[0] - 0
            blocks.append((sick[0], 1))  # one-sided
        
        # Middle blocks (two-sided)
        for i in range(k - 1):
            gap = sick[i + 1] - sick[i] - 1
            if gap > 0:
                blocks.append((gap, 2))  # two-sided
        
        # Rightmost block (if the last infected is not at position n-1)
        if sick[-1] < n - 1:
            # length = (n-1) - sick[-1]
            blocks.append((n - 1 - sick[-1], 1))  # one-sided
        
        # Total number of uninfected children
        # = sum of lengths of all blocks = n - number_of_infected
        M = n - k
        
        # Compute the product of the internal ways for each block.
        # One-sided block of length L => exactly 1 way (infection forced from one end).
        # Two-sided block of length L => 2^(L-1) ways.
        ways_internal = 1
        for length, typ in blocks:
            if length <= 0:
                continue
            if typ == 1:
                # one-sided
                # W(L,1) = 1
                ways_internal = (ways_internal * 1) % MOD
            else:
                # two-sided
                # W(L,2) = 2^(L-1) mod
                ways_internal = (ways_internal * pow(2, length - 1, MOD)) % MOD
        
        # Now multiply by the combinatorial factor for interleaving these blocks.
        # If the blocks have lengths L1, L2, ..., Lr summing to M,
        # the number of ways to interleave them is:
        #   M! / (L1! * L2! * ... * Lr!)
        ways = ways_internal
        ways = (ways * fact[M]) % MOD
        for length, _ in blocks:
            if length > 0:
                ways = (ways * inv_fact[length]) % MOD
        
        return ways % MOD