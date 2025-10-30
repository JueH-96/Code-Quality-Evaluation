class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        import sys
        sys.setrecursionlimit(10**7)
        
        MOD = 10**9 + 7
        
        # We are given "requirements" of the form [end_i, cnt_i],
        # meaning the prefix of length end_i+1 must have exactly cnt_i inversions.
        # There's guaranteed to be at least one requirement where end_i = n - 1,
        # which gives the inversion requirement for the entire permutation.
        #
        # Key idea:
        # 1) Let req[i] = x if there is a requirement that the prefix of length i
        #    must have exactly x inversions; otherwise req[i] = -1. 
        #    Note that prefix of length i means indices [0..i-1], so end_i= e 
        #    corresponds to i = e+1.
        #
        # 2) We'll use a DP approach where dp[i][inv] = number of permutations
        #    of length i (i distinct elements) that have exactly inv inversions.
        #    Here i ranges from 0..n, and inv ranges up to 400 (the max in constraints).
        #
        # 3) The standard recurrence for counting permutations by number of inversions is:
        #    dp[i][inv] = sum of dp[i-1][inv - k] for k = 0..(i-1), if inv >= k
        #    Because when we add the i-th element (0-based counting for the steps),
        #    we can "insert" it in a way that adds k new inversions, where k is how
        #    many of the previously placed elements it is "smaller than".
        #    In code, we usually optimize this with prefix sums so each dp[i][inv]
        #    can be obtained in O(1) from dp[i-1].
        #
        # 4) After computing dp[i][*], if there is a requirement req[i] != -1, 
        #    we keep only dp[i][req[i]] and set dp[i][all other] = 0. 
        #    This enforces the exact inversion count for that prefix length.
        #
        # 5) In the end, because there's guaranteed a requirement for the total length
        #    n, say req[n] = totalInv, the final answer is dp[n][totalInv].
        #
        # 6) One subtlety: a prefix that is not constrained might have more than 400
        #    inversions in theory. However, since all constraints (including the final
        #    one) require ≤ 400 inversions, and inversions do not decrease as the prefix
        #    grows, any prefix that exceeds 400 inversions cannot transition to meet a
        #    later ≤ 400 inversion requirement. Thus restricting dp to 0..400 is safe.

        MAX_INV = 400
        
        # Build an array of length n+1 for the forced inversion count of that prefix length
        # Default -1 means no requirement
        req = [-1] * (n + 1)
        for e, c in requirements:
            req[e + 1] = c
        
        # We must have a requirement for the entire array at index n (since end_i = n-1)
        # If there's no such requirement, the problem specification says it always exists.
        assert req[n] != -1, "By problem statement, there is at least one requirement with end_i = n-1"
        final_inv = req[n]
        
        # dp[i][inv] = number of permutations of length i that have exactly inv inversions
        dp = [[0]*(MAX_INV+1) for _ in range(n+1)]
        dp[0][0] = 1  # 0-length prefix has exactly 0 inversions
        
        for i in range(1, n+1):
            # Compute prefix sums of dp[i-1]
            prefix_sum = [0]*(MAX_INV+1)
            prefix_sum[0] = dp[i-1][0] % MOD
            for inv in range(1, MAX_INV+1):
                prefix_sum[inv] = (prefix_sum[inv-1] + dp[i-1][inv]) % MOD
            
            for inv in range(MAX_INV+1):
                # dp[i][inv] = sum of dp[i-1][inv - k] for k=0..(i-1) if inv>=k
                # In O(1) form:
                # dp[i][inv] = prefix_sum[inv] - prefix_sum[inv - i] (if inv >= i), all modded
                val = prefix_sum[inv]
                if inv - i >= 0:
                    val -= prefix_sum[inv - i]
                dp[i][inv] = val % MOD
            
            # Enforce requirement if any: prefix i must have exactly req[i] inversions
            if req[i] != -1:
                needed = req[i]
                # Zero out everything except dp[i][needed]
                for inv in range(MAX_INV+1):
                    if inv != needed:
                        dp[i][inv] = 0
        
        # The final answer is dp[n][ req[n] ]
        return dp[n][final_inv] % MOD