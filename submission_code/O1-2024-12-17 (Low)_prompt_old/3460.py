class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        import sys
        input_data = sys.stdin.readline  # Not strictly needed in this environment, but good practice
        MOD = 10**9 + 7

        # We will use the classical DP approach for counting permutations by inversion counts:
        # Let dp[m][inv] = number of permutations of length m with exactly inv inversions (0 <= inv <= 400).
        #
        # Recurrence (well-known fact for counting permutations by inversions):
        # dp[m][inv] = sum_{k=0..m-1} dp[m-1][inv - k] , where k <= inv.
        # Explanation: when placing the m-th distinct element (0-based counting) into an already formed permutation
        # of length m-1, if we place it in position k among m slots, we generate k new inversions with the m-1
        # previously placed elements (because it's bigger than everything to its left).
        #
        # Furthermore, we incorporate the prefix constraints:
        # If there's a requirement that the prefix of length (end_i+1) has exactly cnt_i inversions,
        # then after we compute dp[end_i+1][*], we must zero out all dp[end_i+1][j] where j != cnt_i.
        #
        # In the end, since at least one requirement will have end_i = n-1, say it requires cntTot inversions,
        # the answer will be dp[n][cntTot], since that covers permutations of length n with exactly cntTot inversions
        # (and also satisfying all earlier prefix constraints).

        # Step 1: Record prefix constraints. constraints[length] = required_inversions or -1 if none
        constraints = [-1]*(n+1)
        for e, c in requirements:
            constraints[e+1] = c
        
        # We must know the total-inversion requirement for the full permutation (prefix length n).
        # The problem guarantees at least one requirement has end_i = n-1:
        # unify them if there might be more? The problem states "The input is generated such that
        # there is at least one i such that end_i == n - 1." 
        # We only have one such constraint because end_i are unique and there's exactly one prefix with end_i = n-1?
        # The problem does not explicitly say there's only one such requirement, but it does say "The input
        # is generated such that all end_i are unique." Hence indeed there's exactly one constraint with end_i=n-1.
        final_inversions = constraints[n]
        
        # Step 2: Set up DP array
        # dp[m][inv] for 0 <= m <= n, 0 <= inv <= 400
        dp = [[0]*401 for _ in range(n+1)]
        dp[0][0] = 1  # 0-length prefix has 0 inversions in exactly 1 way
        
        # To speed up summation, we'll use prefix sums:
        for m in range(1, n+1):
            # Build prefix sums of dp[m-1][*]
            prefix_sum = [0]*(402)
            run = 0
            for inv in range(401):
                run = (run + dp[m-1][inv]) % MOD
                prefix_sum[inv] = run
            
            # Compute dp[m][inv] via the standard formula using prefix sums:
            # dp[m][inv] = sum_{k=0..m-1} dp[m-1][inv-k], for k <= inv
            # which is prefix_sum[inv] - prefix_sum[inv - m] (handle boundaries)
            for inv in range(401):
                low_idx = inv - (m-1)
                if low_idx <= 0:
                    dp[m][inv] = prefix_sum[inv] % MOD
                else:
                    dp[m][inv] = (prefix_sum[inv] - prefix_sum[low_idx-1]) % MOD
            
            # If there's a constraint for prefix length m, force that constraint
            if constraints[m] != -1:
                required = constraints[m]
                for inv in range(401):
                    if inv != required:
                        dp[m][inv] = 0
        
        # The answer is dp[n][final_inversions] % MOD
        return dp[n][final_inversions] % MOD