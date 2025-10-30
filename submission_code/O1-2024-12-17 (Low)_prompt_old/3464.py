class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        """
        We define the cost of a subarray nums[l..r] (with local indexing from 0)
        as: nums[l] - nums[l+1] + nums[l+2] - ... up to index r (alternating signs),
        i.e. sum_{k=0..(r-l)} nums[l + k] * (-1)^k.

        We wish to split the array (or not split at all) into subarrays to maximize
        the sum of these subarray costs.

        A key observation is that the alternating-sum of a segment [l..r] can be
        related to "prefix alternating sums" of the entire array from index 0 using
        the following idea:

          Let A[i] = nums[0] - nums[1] + nums[2] - ... ± nums[i],
          i.e. A[i] = sum_{k=0..i} [ nums[k] * (-1)^k ], with A[-1] = 0 by convention.

          Then the cost of subarray [l..r] (0-based indices in the global array,
          but starting the sign at + for index l) can be shown to be:
              cost(l, r) = (-1)^l * ( A[r] - A[l-1] ).

        We define dp[i] = the maximum total cost attainable by splitting the subarray
        nums[0..i] into one or more alternating-sum segments.

        It turns out we can maintain dp[i] in O(1) time per step using two running
        "best" values, depending on the parity of the start-index of the last subarray:

            dp[i] = max( A[i] + Me,   # (if the last subarray starts at an even index j)
                         -A[i] + Mo ) # (if the last subarray starts at an odd index j)

        where:
            Me = max_{j even} [ dp[j-1] - A[j-1] ]
            Mo = max_{j odd}  [ dp[j-1] + A[j-1] ]

        As we iterate from left to right, once we compute dp[i], we update Me or Mo
        depending on whether i+1 is even or odd (since i+1 would be the start of the
        next segment).

        Finally, dp[n - 1] is our answer.
        """

        n = len(nums)
        # Edge case: if n == 0, though per the problem constraints n >= 1
        if n == 0:
            return 0
        
        # Build the prefix alternating sums array A, where
        # A[i] = nums[0] - nums[1] + nums[2] - ... ± nums[i]
        A = [0]*n
        A[0] = nums[0]
        for i in range(1, n):
            A[i] = A[i-1] + nums[i] * ((-1)**i)
        
        # dp[i] = max cost splitting up to index i
        dp = [0]*n
        
        # We keep track of:
        # Me = max_{k even} [dp[k-1] - A[k-1]]
        # Mo = max_{k odd}  [dp[k-1] + A[k-1]]
        # with dp[-1] = 0, A[-1] = 0 by convention.
        # Initialize for j = 0 (which is even), so that dp[-1] - A[-1] = 0 is valid
        import math
        Me = 0             # max over even-start subarrays so far
        Mo = -math.inf     # max over odd-start subarrays so far
        
        for i in range(n):
            # dp[i] = max( A[i] + Me, -A[i] + Mo )
            take_even = A[i] + Me    # subarray starts at some even j
            take_odd  = -A[i] + Mo   # subarray starts at some odd j
            dp[i] = max(take_even, take_odd)
            
            # Now update Me or Mo for the next iteration (i+1 is the start of next subarray)
            if (i + 1) % 2 == 0:
                # i+1 is even
                Me = max(Me, dp[i] - A[i])
            else:
                # i+1 is odd
                Mo = max(Mo, dp[i] + A[i])
        
        # The answer is dp[n-1]
        return dp[n-1]