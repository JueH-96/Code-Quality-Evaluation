from typing import List
import math

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        # prefix[i] = sum_{t=0}^{i} (-1)^t * nums[t]
        prefix = [0] * n
        prefix[0] = nums[0]  # (-1)^0 = 1
        for i in range(1, n):
            if i % 2 == 0:
                # even index, factor +1
                prefix[i] = prefix[i-1] + nums[i]
            else:
                # odd index, factor -1
                prefix[i] = prefix[i-1] - nums[i]
        
        # dp[i] = maximum total cost we can achieve for nums[0:i+1]
        dp = [0] * n
        
        # We derived recurrence:
        # Let X = j-1 (where j is the start index of the subarray, with j from 0 to i),
        # then dp[i] = max_{X = -1, 0, 1, ..., i-1} { dp[X] + (-1)^(X+1) * (prefix[i] - prefix[X]) }.
        # For convenience, define A[X] = dp[X] - (-1)^(X+1) * prefix[X], then:
        # dp[i] = max( prefix[i] + max_{X in S0} A[X], -prefix[i] + max_{X in S1} A[X] )
        # where S0 contains indices X with (-1)^(X+1)=1 and S1 contains indices with (-1)^(X+1)=-1.
        #
        # Note: For X = -1, we define dp[-1] = 0 and prefix[-1] = 0, and (-1)^(0) = 1.
        # So initially, best_even = A[-1] = 0. We set best_odd to -infinity.
        
        best_even = 0  # corresponds to indices X with (-1)^(X+1)=1 (X's for which (X+1) is even)
        best_odd = -math.inf  # corresponds to indices X with (-1)^(X+1)=-1 (i.e., (X+1) odd)
        
        for i in range(n):
            # dp[i] = max( prefix[i] + best_even, -prefix[i] + best_odd )
            dp_candidate1 = prefix[i] + best_even
            dp_candidate2 = -prefix[i] + best_odd
            dp[i] = max(dp_candidate1, dp_candidate2)
            
            # After computing dp[i], we treat index X = i as candidate for future transitions.
            # Compute the term to update the appropriate group.
            # determine group by (-1)^(i+1)
            if (i + 1) % 2 == 0:
                # (-1)^(i+1) = 1, so update best_even with A[i] = dp[i] - prefix[i]
                best_even = max(best_even, dp[i] - prefix[i])
            else:
                # (-1)^(i+1) = -1, so update best_odd with A[i] = dp[i] - (-1)*prefix[i] = dp[i] + prefix[i]
                best_odd = max(best_odd, dp[i] + prefix[i])
                
        return dp[n-1]