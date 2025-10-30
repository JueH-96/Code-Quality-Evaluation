from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        """
        We want to make the array nums non-decreasing with the minimum number of operations.
        An operation on nums[i] replaces nums[i] with nums[i] / g, where g is the greatest
        proper divisor of nums[i]. If nums[i] is prime (or 1), its greatest proper divisor is 1,
        and dividing by 1 does not change the value. Hence prime numbers (and 1) cannot be reduced.

        Key Insight:
        1) If x is prime or x=1, it cannot be reduced at all (0 possible "reductions").
        2) If x is composite, exactly one operation can reduce x to its smallest prime factor p
           (since the greatest proper divisor g = x/p, then x / g = p). After that, p is prime
           and cannot be reduced further.

        Therefore, for each element x in nums, we have at most two possible final values:
          - Value A = x  (0 operations)
          - Value B = spf[x],  if spf[x] < x (which happens iff x is composite), costing 1 operation

        We then want to pick a sequence of final values c_0, c_1, ..., c_(n-1) (each c_i is either
        Value A or Value B for nums[i]) such that c_0 <= c_1 <= ... <= c_(n-1), minimizing the
        total cost (sum of operations).

        We can do a simple dynamic programming over i, storing two DP states for each i:
          dpA[i] = minimum cost if we pick c_i = nums[i]     (the "A" choice)
          dpB[i] = minimum cost if we pick c_i = spf[nums[i]] (the "B" choice, if valid)

        Transition:
          Let valA[i] = nums[i], costA[i] = 0
          If spf[nums[i]] < nums[i], let valB[i] = spf[nums[i]], costB[i] = 1, otherwise no B-state.

          dpA[i] = min over dpA[i-1] or dpB[i-1], subject to the non-decreasing condition
                    (chosen_value_{i-1} <= valA[i]), plus costA[i].
          dpB[i] = similarly, if valB[i] exists, from dpA[i-1] or dpB[i-1] and chosen_value_{i-1} <= valB[i].

        Finally, the answer is min(dpA[n-1], dpB[n-1]). If both are infinite, return -1.

        Complexity:
        - We precompute spf up to 1e6 using a sieve in O(N log log N) time, with N=1e6.
        - Then we do a single pass DP in O(n), with n up to 1e5. This is efficient enough.
        """

        # Precompute the smallest prime factor (spf) up to 1,000,000
        # We'll do it once, stored in a static/class-level array if needed.
        # For clarity in a single function, we'll compute it here (caching if desired).
        MAXV = 10**6
        # If we have done it before, we might skip, but let's do it once:
        if not hasattr(Solution, "_spf"):
            spf = [0]*(MAXV+1)
            spf[1] = 1
            for i in range(2, MAXV+1):
                if spf[i] == 0:  # i is prime
                    spf[i] = i
                    if i*i <= MAXV:
                        for j in range(i*i, MAXV+1, i):
                            if spf[j] == 0:
                                spf[j] = i
            Solution._spf = spf
        else:
            spf = Solution._spf

        n = len(nums)
        if n == 1:
            # A single element is trivially non-decreasing, no operations needed.
            return 0

        # Prepare arrays for valA, costA, valB, costB
        valA = [0]*n
        costA = [0]*n
        valB = [None]*n
        costB = [None]*n

        for i in range(n):
            x = nums[i]
            valA[i] = x
            costA[i] = 0
            # Check if x can be reduced:
            p = spf[x]  # smallest prime factor
            if p < x:
                # Then we can reduce x by 1 operation down to p
                valB[i] = p
                costB[i] = 1
            else:
                # x is prime or x=1 => no further "B" possibility
                valB[i] = None
                costB[i] = None

        # We'll keep dpA[i], dpB[i] for the minimal cost, or a large INF if impossible
        INF = 10**15
        dpA = [INF]*n
        dpB = [INF]*n

        # Initialize for i=0
        dpA[0] = costA[0]
        if valB[0] is not None:
            dpB[0] = costB[0]
        else:
            dpB[0] = INF

        # Fill DP
        for i in range(1, n):
            # Reset dp values for i
            bestA_prev = dpA[i-1]
            bestB_prev = dpB[i-1]

            # dpA[i] = INF by default
            dpA[i] = INF
            dpB[i] = INF

            # If dpA[i-1] is not INF and valA[i-1] <= valA[i], we can transition
            if bestA_prev != INF and valA[i-1] <= valA[i]:
                cand = bestA_prev + costA[i]
                if cand < dpA[i]:
                    dpA[i] = cand

            # If dpB[i-1] is not INF and valB[i-1] <= valA[i], we can transition
            if bestB_prev != INF and valB[i-1] is not None and valB[i-1] <= valA[i]:
                cand = bestB_prev + costA[i]
                if cand < dpA[i]:
                    dpA[i] = cand

            # Now for dpB[i], if valB[i] is not None
            if valB[i] is not None:
                # Check transitions to dpB[i]
                if bestA_prev != INF and valA[i-1] <= valB[i]:
                    cand = bestA_prev + costB[i]
                    if cand < dpB[i]:
                        dpB[i] = cand
                if bestB_prev != INF and valB[i-1] is not None and valB[i-1] <= valB[i]:
                    cand = bestB_prev + costB[i]
                    if cand < dpB[i]:
                        dpB[i] = cand

            # If both dpA[i] and dpB[i] remain INF, it means no valid non-decreasing arrangement
            # can reach index i. We can detect early fail if desired, but let's just continue;
            # if it stays INF, the final answer will be -1.

        ans = min(dpA[n-1], dpB[n-1])
        return ans if ans < INF else -1