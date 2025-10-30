class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        """
        We want the number of permutations of [0..n-1] that satisfy:
           For each requirement [end_i, cnt_i], the prefix perm[:end_i+1]
           has exactly cnt_i inversions.
        At least one requirement has end_i = n-1, so the entire permutation
        must have exactly that many inversions overall.
        
        Key idea:
         • A classic fact is that the number of permutations of length k (with labels 1..k)
           having exactly inv inversions can be counted via a well-known DP:
             dp(k, inv) = sum of dp(k-1, inv - x) for x = 0..(k-1),
           where dp(1,0)=1 and dp(1,x>0)=0, etc.
         • However, here we also have "prefix constraints" for various prefix lengths < n.
           Interpreted literally, a prefix constraint says: "the leftmost k elements
           of the final permutation must have exactly need[k] inversions."
         • A well-known observation is that any set of k distinct numbers, when viewed
           only by their relative order, is isomorphic to a permutation of {1..k}.
           The number of inversions within that prefix depends only on the relative order,
           not on which actual subset of [0..n-1] was chosen.
         • Therefore, to incorporate prefix constraints "prefix of length k has cnt inversions,"
           we can do a DP over k from 0..n, counting how many ways to form an "order-isomorphic"
           sequence of length k with the required number of inversions.  When we move from
           k to k+1, we add one more "largest label," and placing that largest label among
           the previous k positions can add between 0 and k new inversions.
         • At each prefix length k for which there's a constraint, we enforce that dp(k, x)=0
           unless x == need[k].
         • In the end, dp(n, need[n]) gives the total count of permutations of size n
           (labels 1..n in relative order) that satisfy exactly need[k] inversions at
           each constrained prefix k.  This count matches the count of permutations of
           [0..n-1] that meet the same prefix-inversion constraints, because inversions
           depend only on relative ordering.
        
        Steps:
          1) Build an array "need" of length n+1, initialized to -1.  For each [end_i, cnt_i],
             set need[end_i+1] = cnt_i.
          2) Let dp[k][inv] = number of ways to form a length-k arrangement (labels 1..k)
             with exactly inv inversions, subject to already satisfying the prefix constraints
             for lengths <= k.
          3) Initialize dp[0][0] = 1, and dp[0][inv>0] = 0.
          4) For k from 0 to n-1:
               - If need[k] != -1 and k > 0, then enforce dp[k] by zeroing out
                 all dp[k][inv] except dp[k][need[k]].
               - Compute dp[k+1][inv'] by the recurrence:
                   dp[k+1][inv'] = sum_{j = inv' - k .. inv'} dp[k][j],
                 capped at j >= 0, and inv' up to 400.  This can be done in O(1) per inv' with
                 prefix sums.
               - If need[k+1] != -1, then enforce dp[k+1] similarly.
          5) Finally, the answer is dp[n][ need[n] ] (there is guaranteed a constraint
             need[n] != -1 in the input).
          6) Return the result modulo 10^9+7.
        """
        MOD = 10**9 + 7
        
        # Step 1: Build the "need" array to store required inversions for each prefix length
        need = [-1]*(n+1)
        for end_idx, cnt in requirements:
            need[end_idx+1] = cnt
        
        # Step 2: dp array, dp[k][inv] = count of valid orderings of length k with inv inversions
        dp = [[0]*401 for _ in range(n+1)]
        dp[0][0] = 1  # empty arrangement has 0 inversions
        
        for k in range(n+1):
            # Enforce a prefix constraint for length k, if any (skip k=0, as there's no real "prefix 0" constraint)
            if k > 0 and need[k] != -1:
                required = need[k]
                for inv in range(401):
                    if inv != required:
                        dp[k][inv] = 0
            
            # If we've reached k == n, we don't build dp[n+1], so stop
            if k == n:
                break
            
            # Step 4: Compute dp[k+1] from dp[k]
            # Make a prefix-sum array S so that S[i+1] = sum_{x=0..i} dp[k][x]
            S = [0]*(402)
            for i in range(401):
                S[i+1] = (S[i] + dp[k][i]) % MOD
            
            for inv_next in range(401):
                # dp[k+1][inv_next] = sum of dp[k][j] for j in [inv_next - k .. inv_next], intersect [0..400]
                low = inv_next - k
                if low < 0:
                    low = 0
                # sum_{j=low..inv_next} dp[k][j] = S[inv_next+1] - S[low]
                val = S[inv_next+1] - S[low]
                dp[k+1][inv_next] = val % MOD
        
        # After finishing, we must have a constraint for prefix length n:
        # The final answer is dp[n][need[n]] mod 1e9+7
        total_inversions = need[n]  # guaranteed not -1 by the problem statement
        return dp[n][total_inversions] % MOD