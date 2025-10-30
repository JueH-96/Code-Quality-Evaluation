class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        """
        We interpret the cost of a subarray nums[l..r] as the sum of its elements
        in alternating signs, starting with plus for nums[l].  Formally,
            cost(l, r) = nums[l] - nums[l+1] + nums[l+2] - ...  (local indexing)
        We want to split the entire array into contiguous subarrays so that the
        sum of the costs of those subarrays is maximized.

        A classic way to handle 'local alternating signs' is to re-express cost(l, r)
        in terms of a prefix sum that uses global alternating signs, then adjust
        for the subarray's start.  One can show:

            cost(l, r) = (-1)^l * [ (prefix_alternating[r]) - (prefix_alternating[l-1]) ],

        if prefix_alternating[i] = nums[0] * (+/-) + ... + nums[i] * (+/-)
        with a global pattern +/- determined by the index.  However, to reset
        signs at subarray boundaries (i.e. always start with + within each subarray),
        we factor out (-1)^l appropriately.

        To compute the maximum split cost in O(n), we use a DP approach:
            dp[i] = maximum total cost when we split up to index i.
        Then
            dp[i] = max over j in [0..i] of [ dp[j-1] + cost(j, i) ].
        With the prefix sums trick, "cost(j, i)" can be done in O(1).  Rearranging
        leads to a neat formula that allows a running "bestEven" and "bestOdd" to
        capture the needed maxima.  The result is dp[n-1].

        Steps:
          1) Build prefix2 such that prefix2[k] = sum_{m=0..k-1} [ nums[m]*((-1)^m ) ].
          2) Maintain:
                bestEven = max of [dp[j-1] - prefix2[j]*(+1)] for even j
                bestOdd  = max of [dp[j-1] - prefix2[j]*(-1)] for odd j
             because cost(j, i) = (-1)^j * ( prefix2[i+1] - prefix2[j] ).
          3) Then:
                dp[i] = max( prefix2[i+1] + bestEven,   # if we pick j even
                             -prefix2[i+1] + bestOdd )  # if we pick j odd
          4) Update bestEven or bestOdd using dp[i] afterward for next iteration.

        This runs in O(n) time and covers the possibility of no splits (j=0 => the
        entire segment from 0..i), as well as any number of splits in between.
        """
        import math

        n = len(nums)
        if n == 0:
            return 0
        # prefix2[i] = sum of nums[0]*(-1)^0 + nums[1]*(-1)^1 + ... + nums[i-1]*(-1)^(i-1)
        prefix2 = [0]*(n+1)
        for i in range(n):
            if i % 2 == 0:  # even index => sign = +
                prefix2[i+1] = prefix2[i] + nums[i]
            else:           # odd index => sign = -
                prefix2[i+1] = prefix2[i] - nums[i]

        # dp[i] = max sum of costs splitting up to index i
        dp = [0]*n
        
        # bestEven and bestOdd track the best values for "dp[j-1] - prefix2[j]*(-1)^j"
        bestEven = 0    # when j=0, dp[-1]=0, prefix2[0]=0 => T_0=0
        bestOdd = -math.inf
        
        for i in range(n):
            # pick j with even j => cost = + prefix2[i+1]
            # pick j with odd  j => cost = - prefix2[i+1]
            candidate_even = prefix2[i+1] + bestEven
            candidate_odd  = -prefix2[i+1] + bestOdd
            dp[i] = max(candidate_even, candidate_odd)
            
            # Now update bestEven/bestOdd for the next index:
            # T_{i+1} = dp[i] - prefix2[i+1]*((-1)^(i+1))
            if (i+1) % 2 == 0:  # even
                T = dp[i] - prefix2[i+1]  # (-1)^(i+1) = +1 if (i+1) even
                bestEven = max(bestEven, T)
            else:
                T = dp[i] + prefix2[i+1]  # (-1)^(i+1) = -1 if (i+1) is odd => subtract => plus prefix2
                bestOdd = max(bestOdd, T)

        return dp[n-1]