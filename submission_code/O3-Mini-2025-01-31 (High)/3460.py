from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        mod = 10**9 + 7
        MAX = 400  # we only care about inversion counts up to 400 because every requirement has cnt <= 400

        # Build an array "cons" for required inversion count on a prefix.
        # In our formulation the “prefix” length is counted as the number of elements.
        # A requirement [end, cnt] means that prefix of length (end+1) must have inversion total cnt.
        cons = [-1] * n
        for end, cnt in requirements:
            cons[end] = cnt

        # We use the well–known bijection between permutations and inversion sequences:
        # For a permutation of length n there is a unique sequence a[0],a[1],...,a[n-1] such that
        # 0 <= a[i] <= i   for 0 <= i < n, and the total number of inversions is sum(a[i]).
        # Moreover, the number of permutations with inversion count = k equals the number of such sequences with sum=k.
        #
        # We will do a DP where dp[L][s] = number of ways to choose an inversion sequence for the first L positions
        # (that is, for a permutation of length L) with total inversion sum equal to s.
        # The recurrence is:
        #   dp[1][0] = 1   (a[0] must be 0)
        #   dp[L+1][s] = sum_{r=0}^{L} dp[L][s - r]   for L >= 1   because for the (L+1)-th element,
        #   the allowed a[L] can be any r in 0..L.
        #
        # Also, if a requirement forces the inversion total on a prefix, we “zero‐out” all dp states that
        # do not match the required value.
        #
        # We only need to count inversion totals up to MAX.
        
        # Initialize dp for a permutation of length 1.
        # dp[1] is an array of length MAX+1 (0 through MAX); only dp[0]=1 is nonzero.
        dp = [0] * (MAX + 1)
        dp[0] = 1
        # Check if there is a constraint for the prefix of length 1 (i.e. requirement for end index 0).
        # In any one–element permutation the inversion count is always 0.
        if cons[0] != -1 and cons[0] != 0:
            return 0

        # Now, iterate from the current permutation length L = 1 to final length n.
        # When we add a new element (position L, so new length L+1) the new inversion value (the next term in the inversion sequence)
        # can be any r from 0 to L.
        # So our recurrence becomes:
        #   new_dp[s] = sum_{r=0}^{L} dp[s - r]   for s in 0..MAX.
        #
        # We optimize the summation by working with prefix sums.
        for L in range(1, n):  # current length is L; we will build dp for length L+1 (which corresponds to prefix index L)
            new_dp = [0] * (MAX + 1)
            
            # Compute prefix sums from dp.
            # Let P[x] = dp[0] + dp[1] + ... + dp[x] (all modulo mod).
            P = [0] * (MAX + 1)
            P[0] = dp[0] % mod
            for s in range(1, MAX + 1):
                P[s] = (P[s - 1] + dp[s]) % mod

            # Transition: for each new total inversion count s (0 <= s <= MAX),
            # dp[L+1][s] = sum_{r=0}^{L} dp[s - r] (only when s - r >= 0).
            # Using the prefix-sum array P, we write:
            #   If s < L+1: new_dp[s] = P[s]
            #   Else:      new_dp[s] = P[s] - P[s - (L + 1)]
            for s in range(0, MAX + 1):
                if s < L + 1:
                    new_dp[s] = P[s] % mod
                else:
                    new_dp[s] = (P[s] - P[s - (L + 1)]) % mod

            # If there is a requirement for the prefix ending at index L (recall: requirement for prefix length L+1)
            # then force dp[L+1][s] to be 0 unless s equals that required inversion count.
            if cons[L] != -1:
                req = cons[L]
                val = new_dp[req] % mod
                new_dp = [0] * (MAX + 1)
                if req <= MAX:
                    new_dp[req] = val
            
            dp = new_dp

        # At the end, dp now represents the count for a permutation of length n (prefix ending at index n-1).
        # By the input guarantee, there is at least one requirement with end == n - 1,
        # so we have cons[n-1] != -1 and the state will have nonzero count only at that inversion value.
        if cons[n - 1] != -1:
            return dp[cons[n - 1]] % mod
        else:
            return sum(dp) % mod