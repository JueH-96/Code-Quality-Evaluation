class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        import sys
        input_data = sys.stdin.read()  # Not actually needed in the LeetCode environment;
                                       # here for completeness if run elsewhere.
        
        MOD = 10**9 + 7
        
        # ----------------------------------------------------------------
        # 1) We will interpret "prefix of length i" to mean the first i elements
        #    of the permutation (i.e. indices 0..i-1).  In the problem statement,
        #    a requirement [end_i, cnt_i] means that perm[0..end_i] (inclusive)
        #    has cnt_i inversions.  The length of that prefix is (end_i + 1).
        #
        # 2) We know at least one requirement has end_i = n - 1, which gives
        #    us the inversion count for the entire permutation of length n.
        #
        # 3) We will use a classic DP approach that counts the number of permutations
        #    of length i with exactly k inversions (for k up to 400).  The well-known
        #    formula/recurrence is:
        #       DP[i+1][k] = sum_{r=0..i} DP[i][k-r]
        #    because when we add one more element (the (i+1)-th distinct value),
        #    we can place it in any of (i+1) positions in the new permutation of length i+1,
        #    which introduces r new inversions for r in [0..i].
        #
        # 4) However, we also have to enforce the constraints that whenever i+1
        #    equals (end_i+1) for some requirement, the number of inversions must
        #    be exactly cnt_i.  We will store these constraints in an array req_map
        #    of size (n+1).  If req_map[length] = c != -1, it means that any permutation
        #    of length "length" must have exactly c inversions in our DP.  We will
        #    "zero out" the DP counts that do not match c at that stage.
        #
        # 5) At the end, because one requirement must have end_i = n - 1, we know
        #    req_map[n] = the required total inversion count for the full permutation.
        #    Our answer is simply DP[n][ req_map[n] ].
        #
        # 6) We will do all DP calculations modulo 10^9+7, and we only keep track
        #    of inversion counts up to 400 (the maximum possible cnt_i).
        #
        # This method is often referred to as the "polynomial (or 1D) DP for counting
        # permutations by inversion number," with an additional "filtering" step each
        # time we reach a prefix length constrained by the input.
        # ----------------------------------------------------------------

        # Build the req_map array of length n+1.  req_map[i] = -1 means no constraint.
        # If we see a requirement [end_i, cnt_i], that corresponds to a prefix length
        # of (end_i+1), so we store req_map[end_i+1] = cnt_i.
        
        req_map = [-1] * (n + 1)
        # Read the requirements into req_map
        for e, c in requirements:
            req_map[e + 1] = c
        
        # We will maintain DP for i = 0..n, and for inv = 0..400.
        # To save memory, we can alternate between two rows: dp_current and dp_next.
        
        max_inv = 400  # given in the problem constraints
        
        dp_current = [0] * (max_inv + 1)
        dp_current[0] = 1  # For length 0, there's exactly 1 "empty" permutation with 0 inversions.
        
        for length in range(n):
            # length goes 0..(n-1).  We'll compute dp for (length+1).
            dp_next = [0] * (max_inv + 1)
            
            # Build prefix sums of dp_current for fast convolution.
            # prefix_sum[k] = sum of dp_current[0..k], inclusive.
            prefix_sum = [0] * (max_inv + 2)
            for k in range(max_inv + 1):
                prefix_sum[k + 1] = (prefix_sum[k] + dp_current[k]) % MOD
            
            # Recurrence:
            # dp_next[k] = sum_{r = 0..length} dp_current[k-r], as long as k-r >= 0.
            # That can be done with prefix sums:
            # dp_next[k] = prefix_sum[k+1] - prefix_sum[k - length], adjusting bounds.
            for k in range(max_inv + 1):
                # r can go up to length, so k-r >= 0 => r <= k.
                # effectively r in [0..min(length, k)], but we can do:
                # dp_next[k] = prefix_sum[k+1] - prefix_sum[k - length] (if k-length >= 0)
                
                # The lower index for prefix_sum is k - length
                low = k - length
                if low < 0:
                    # sum all dp_current[0..k]
                    val = prefix_sum[k + 1]
                else:
                    # prefix_sum[k+1] - prefix_sum[low]
                    val = prefix_sum[k + 1] - prefix_sum[low]
                
                dp_next[k] = val % MOD
            
            # Now, if there's a requirement at prefix length = length+1,
            # we must force dp_next[k] = 0 unless k == req_map[length+1].
            req_val = req_map[length + 1]
            if req_val != -1:
                # Zero out everything except dp_next[req_val]
                keep = dp_next[req_val] if req_val <= max_inv else 0
                for k in range(max_inv + 1):
                    dp_next[k] = 0
                if req_val <= max_inv:
                    dp_next[req_val] = keep % MOD
            
            dp_current = dp_next
        
        # Finally, we know one of the requirements must be for the full length n.
        # Let total_inv = req_map[n].  That is the required number of inversions for
        # the entire permutation.  Our answer is dp_current[ total_inv ].
        
        total_inv = req_map[n]
        # If for some reason total_inv == -1, that would mean there's no constraint
        # on the full permutation, but the problem states there is at least one
        # requirement with end_i = n-1, so total_inv should not be -1.
        if total_inv < 0 or total_inv > max_inv:
            return 0
        
        return dp_current[total_inv] % MOD