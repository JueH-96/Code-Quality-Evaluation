class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        import sys
        sys.setrecursionlimit(10**7)
        from functools import lru_cache

        MOD = 10**9 + 7

        # Convert binary string to integer
        n = int(s, 2)
        if n == 1:
            # No positive integers less than 1
            return 0

        # If k == 5, then every positive integer < n is 5-reducible,
        # because any number up to 2^800 reduces to 1 in at most 4 steps.
        if k == 5:
            return (n - 1) % MOD

        # Precompute the number of steps needed to reduce t to 1 (chain length),
        # for t up to 800. (popcount(t) <= 800 for any number < 2^800)
        chain_length_cache = [-1] * 801
        chain_length_cache[1] = 0

        @lru_cache(None)
        def chain_len(t: int) -> int:
            if chain_length_cache[t] != -1:
                return chain_length_cache[t]
            if t == 1:
                return 0
            # Count set bits in t
            c = bin(t).count('1')
            chain_length_cache[t] = 1 + chain_len(c)
            return chain_length_cache[t]

        # Fill the cache
        for val in range(2, 801):
            if chain_length_cache[val] == -1:
                chain_len(val)

        # For an integer x > 1, it is k-reducible if:
        #   1 + chain_length(popcount(x)) <= k,
        # i.e. chain_length(popcount(x)) <= k - 1.
        # We'll mark which popcounts are allowed.
        is_allowed = [False] * 801
        for t in range(1, 801):
            if chain_length_cache[t] <= k - 1:
                is_allowed[t] = True

        # We want to count how many x in [0..(n-1)] have popcount(x) in the allowed set.
        # Then we exclude 0 automatically because popcount(0)=0 not in [1..800].
        m = n - 1
        m_bin = bin(m)[2:]
        L = len(m_bin)

        @lru_cache(None)
        def dp(pos: int, ones: int, smaller: bool) -> int:
            # pos: current index in m_bin (0-based, left to right)
            # ones: how many 1-bits chosen so far
            # smaller: have we already chosen a prefix < the prefix of m_bin?
            if ones > 800:
                return 0  # no need to track further, popcount can't exceed 800
            if pos == L:
                return 1 if is_allowed[ones] else 0

            limit = int(m_bin[pos]) if not smaller else 1
            total = 0
            for bit in range(limit + 1):
                total += dp(pos + 1,
                            ones + bit,
                            smaller or (bit < limit))
            return total % MOD

        return dp(0, 0, False) % MOD