class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        """
        We want the number of binary arrays with exactly 'zero' zeroes and 'one' ones,
        such that no subarray of length > limit is all 0's or all 1's.
        
        Equivalently, we cannot have more than 'limit' consecutive 0's or 1's.
        
        A well-known way to count the sequences that have exactly zero 0's, one 1's, and
        no run of 0's or 1's exceeding 'limit' is to realize that the sequence can be seen
        as an alternating chain of "blocks" of 0's and 1's, each block size at most 'limit'.
        
        Concretely:
        1) The array can start with 0 or start with 1.
        2) Suppose the total number of blocks is k. Then:
           - If the array starts with 0, we have (k+1)//2 blocks of 0's and k//2 blocks of 1's.
           - If the array starts with 1, we have k//2 blocks of 0's and (k+1)//2 blocks of 1's.
        3) We must partition 'zero' zeros into those blocks (each block >= 1 and <= limit)
           and 'one' ones into their blocks (each block >= 1 and <= limit).
        
        Let dpC[n][k] = number of ways to write n as a sum of k positive integers, each ≤ limit.
        Then the answer is the sum over all k from 2..(zero+one) of:
          dpC[zero][(k+1)//2] * dpC[one][k//2]   (start with 0)
        + dpC[zero][k//2]     * dpC[one][(k+1)//2] (start with 1)
        
        We compute dpC efficiently using prefix sums in O(N*K) time, where N, K ≤ 200.
        """
        MOD = 10**9 + 7
        N = max(zero, one)  # largest number of bits we might partition
        
        # Build dpC array:
        # dpC[n][k] = number of ways to sum n as k positive parts, each <= limit.
        dpC = [[0]*(N+1) for _ in range(N+1)]
        dpC[0][0] = 1  # 0 can be written as 0 parts in exactly 1 way
        
        for k in range(1, N+1):
            # prefix sums for dpC[..][k-1]
            prefix = [0]*(N+1)
            run_sum = 0
            for x in range(N+1):
                run_sum = (run_sum + dpC[x][k-1]) % MOD
                prefix[x] = run_sum
            
            for nval in range(N+1):
                if nval == 0:
                    dpC[nval][k] = 0  # with k>0 parts, can't sum to 0 (unless k=0)
                else:
                    # dpC[nval][k] = sum_{i=1..limit} of dpC[nval-i][k-1], i <= nval
                    val = prefix[nval-1]  # sum of dpC[0..nval-1][k-1]
                    if nval - limit - 1 >= 0:
                        val = (val - prefix[nval - limit - 1]) % MOD
                    dpC[nval][k] = val % MOD
        
        # Now compute the total number of stable arrays
        total_length = zero + one
        ans = 0
        for k in range(2, total_length + 1):
            # Case 1: start with 0
            zBlocks = (k + 1) // 2
            oBlocks = k // 2
            if zBlocks <= zero and oBlocks <= one:
                ans = (ans + dpC[zero][zBlocks] * dpC[one][oBlocks]) % MOD
            
            # Case 2: start with 1
            zBlocks = k // 2
            oBlocks = (k + 1) // 2
            if zBlocks <= zero and oBlocks <= one:
                ans = (ans + dpC[zero][zBlocks] * dpC[one][oBlocks]) % MOD
        
        return ans % MOD