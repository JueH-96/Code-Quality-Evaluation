class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        """
        We interpret the problem (consistent with the examples) as follows:
        
        - When we take a subarray nums[l..r], its cost is computed by alternating
          plus and minus starting with plus at index l. That is:
              cost(l, r) = nums[l] - nums[l+1] + nums[l+2] - nums[l+3] + ...  (up to r)
          in other words, for i from l to r (inclusive), we add
              ((-1)^(i - l)) * nums[i].
        
        - We may split the entire array into one or more subarrays so as to maximize
          the sum of their costs. Each subarray independently "resets" the alternating
          signs starting with plus at the left endpoint.
        
        EXAMPLE CHECKS:
         1) nums = [1, -2, 3, 4]
            - If not split, the cost from 0..3 is 1 - (-2) + 3 - 4 = 2.
            - If we split as [1, -2, 3] and [4]:
                cost(0..2) = 1 - (-2) + 3 = 1 + 2 + 3 = 6
                cost(3..3) = 4
                total = 10, which matches the example output.
        
         2) nums = [1, -1, 1, -1]
            - Not split: cost(0..3) = 1 - (-1) + 1 - (-1) = 1 + 1 + 1 + 1 = 4
            - That alone gives 4, which is indeed the maximum. The example also shows splitting
              into two [1, -1] and [1, -1] subarrays yields (1 - (-1)) + (1 - (-1)) = 2 + 2 = 4.
        
        A KEY OBSERVATION FOR FAST DP:
        
        Define an alternating-sum prefix S[i] = sum_{k=0..i} [ ((-1)^k) * nums[k] ].
        Then the cost of a subarray nums[l..r] (with alternating signs restarting at l)
        can be written as:
            cost(l, r) = sum_{i=l..r} [ ((-1)^(i - l)) * nums[i] ]
                       = ((-1)^l) * [ S[r] - S[l-1] ]   (with S[-1] = 0 by convention)
                       =  (+1 or -1) * [ partial difference in S[...] ] depending on l's parity
        
        We let dp[i] = maximum achievable cost by splitting exactly up to index i.
        
        Recurrence:
          dp[i] = max_{all valid k <= i} [ dp[k-1] + cost(k, i) ]
                 = max_{k=0..i}
                     dp[k-1] + ((-1)^k)*( S[i] - S[k-1] )
          (where dp[-1] := 0, S[-1] := 0)
        
        Expand that term:
          cost(k, i) = ((-1)^k)*( S[i] - S[k-1] )
            - If k is even, ((-1)^k) = +1, so cost(k, i) = S[i] - S[k-1].
            - If k is odd,  ((-1)^k) = -1, so cost(k, i) = -(S[i] - S[k-1]) = S[k-1] - S[i].
        
        Hence
          dp[i] = max(
             max_{k even} [ dp[k-1] + (S[i] - S[k-1]) ],
             max_{k odd}  [ dp[k-1] + (S[k-1] - S[i]) ]
          )

        We can keep track in O(1) of these best values:
          M_even = max_{k even} [ dp[k-1] - S[k-1] ]   (so that adding S[i] yields dp[k-1] + S[i] - S[k-1])
          M_odd  = max_{k odd}  [ dp[k-1] + S[k-1] ]   (so that subtracting S[i] yields dp[k-1] + S[k-1] - S[i])

        Then
          dp[i] = max( S[i] + M_even, -S[i] + M_odd )

        After we compute dp[i], we update M_even or M_odd depending on the parity of the next starting index (i+1):
          - If (i+1) is even, that means k = i+1 is even for the next subarray start,
            so we might then want to update M_even using dp[i] - S[i].
          - If (i+1) is odd, we update M_odd using dp[i] + S[i].

        Finally, dp[n-1] is our answer.

        This runs in O(n) time and matches all the given examples.
        """

        n = len(nums)
        if n == 0:
            return 0  # though by constraints we always have at least 1 element

        # S[i] = sum of alternating-signed nums up to index i (0-based),
        #        i.e. nums[0] - nums[1] + nums[2] - nums[3] + ...
        S = [0] * n
        S[0] = nums[0]
        for i in range(1, n):
            if i % 2 == 1:
                S[i] = S[i-1] - nums[i]
            else:
                S[i] = S[i-1] + nums[i]

        dp = [0]*n  # dp[i] = max achievable cost using subarray splits up to index i
        # We'll track:
        # M_even = max of dp[k-1] - S[k-1] over even k
        # M_odd  = max of dp[k-1] + S[k-1] over odd k
        # Initially, dp[-1] = 0 and S[-1] = 0 by convention
        # so for k=0 (which is even), "dp[-1] - S[-1]" = 0 - 0 = 0. That initializes M_even.
        M_even = 0
        M_odd = float('-inf')

        for i in range(n):
            # Compute dp[i]
            # dp[i] = max( S[i] + M_even, -S[i] + M_odd )
            candidate_even = S[i] + M_even   # subarray start was even
            candidate_odd =  -S[i] + M_odd   # subarray start was odd
            dp[i] = max(candidate_even, candidate_odd)
            
            # Update M_even/M_odd for the next index (i+1) as a potential start
            if (i + 1) % 2 == 0:  # i+1 even => update M_even
                M_even = max(M_even, dp[i] - S[i])
            else:  # i+1 odd => update M_odd
                M_odd = max(M_odd, dp[i] + S[i])

        return dp[n-1]