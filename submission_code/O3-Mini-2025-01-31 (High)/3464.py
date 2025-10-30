from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        # We want to partition the array into subarrays.
        # For any subarray starting at index j and ending at index i,
        # its cost is given by:
        #    cost(j, i) = ∑[k=j to i] (-1)^(k-j) * nums[k]
        #
        # Notice that we can rewrite:
        #    cost(j, i) = (-1)^j * (∑[k=j to i] (-1)^k * nums[k])
        #
        # Let alt[i] = ∑[m=0 to i] (-1)^m * nums[m] with alt[-1] defined as 0.
        # Then the cost for subarray starting at j becomes:
        #    cost(j, i) = (-1)^j * (alt[i] - alt[j-1])
        #
        # Let dp[i] be the maximum total cost we can achieve for the prefix nums[0..i].
        # If the last subarray of an optimal partition starts at index j (0 <= j <= i),
        # then
        #    dp[i] = dp[j-1] + (-1)^j * (alt[i] - alt[j-1]).
        #
        # Rearranging, we can write for every valid j:
        #    dp[i] = (-1)^j * alt[i] + (dp[j-1] - (-1)^j * alt[j-1])
        #
        # Define for each possible starting index j (of the last subarray) a candidate value:
        #    X[j] = dp[j-1] - (-1)^j * alt[j-1].
        # Then, for any i (i >= 0),
        #    dp[i] = max_{0 <= j <= i} { (-1)^j * alt[i] + X[j] }.
        #
        # We can split the maximization based on the parity of j:
        #  • If j is even, then (-1)^j = +1 so candidate = alt[i] + (dp[j-1] - alt[j-1]),
        #  • If j is odd, then (-1)^j = -1 so candidate = -alt[i] + (dp[j-1] + alt[j-1]).
        #
        # So, if we maintain:
        #    best_even = max_{j even, j <= current position} (dp[j-1] - alt[j-1])
        #    best_odd  = max_{j odd, j <= current position} (dp[j-1] + alt[j-1])
        # (with the convention that for j = 0, dp[-1] = 0 and alt[-1] = 0, so X[0] = 0)
        #
        # Then:
        #    dp[i] = max( alt[i] + best_even,  -alt[i] + best_odd )
        #
        # And after computing dp[i], we update the candidate for being a starting index for a segment that
        # might begin at j = i+1. Note that the parity for index j = i+1 is determined by (i+1) mod 2.
        #
        # To implement this in one pass we will:
        #  - Compute alt on the fly: alt_i = alt_{i-1} + (-1)^i * nums[i]
        #  - Compute dp[i] using the best values for even and odd startups.
        #  - Update best for j = i+1:
        #         if (i+1) is even, then candidate = dp[i] - alt[i]
        #         if (i+1) is odd,  then candidate = dp[i] + alt[i]
        
        n = len(nums)
        # Use a very low integer for -infinity; numbers are at most around 1e14, so this is safe.
        NEG_INF = -10**20  
        
        # best_even will be the maximum candidate among segment start indices with even index.
        # Initially, when j=0 the candidate is dp[-1] - (-1)^0 * alt[-1] = 0.
        best_even = 0  
        best_odd = NEG_INF  # No candidate from an odd starting index initially.
        
        alt = 0  # This will hold alt[i] = sum_{m=0}^{i} (-1)^m * nums[m]
        dp = 0  # Will store dp[i] for the current index i
        
        for i in range(n):
            # Compute the alternating-sum prefix for current index.
            # sign = 1 if i is even, -1 if odd.
            sign = 1 if (i & 1) == 0 else -1
            alt += sign * nums[i]
            
            # dp[i] equals the best option among segments that started at an even index j
            # and those that started at an odd index j.
            # For j even: candidate value = alt + best_even.
            # For j odd: candidate value = -alt + best_odd.
            dp = max(alt + best_even, -alt + best_odd)
            
            # Now update the candidate for a new segment starting at index j = i+1.
            # The term to update is: X[j] = dp[i] - (-1)^(i+1) * alt.
            if ((i + 1) & 1) == 0:  # (i+1) is even ⇒ (-1)^(i+1) = +1.
                candidate = dp - alt
                best_even = max(best_even, candidate)
            else:  # (i+1) is odd ⇒ (-1)^(i+1) = -1.
                candidate = dp + alt
                best_odd = max(best_odd, candidate)
        
        return dp