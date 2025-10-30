class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[i]: maximum total cost using elements 0..i
        dp = [0] * n
        
        # P[i] = sum_{k=0}^{i} (-1)^k * nums[k]
        P = [0] * n
        
        # For ease, define a function to get the sign: sign(i)=1 if i even, -1 if odd.
        def sign(i: int) -> int:
            return 1 if i % 2 == 0 else -1
        
        # Initialize P[0]
        P[0] = sign(0) * nums[0]
        # For dp[0], there is only one segment starting at index 0:
        # dp[0] = cost(0,0) = nums[0] (because sign(0)=+1)
        dp[0] = nums[0]
        
        # bestEven: maximum candidate among L even for dp[L-1] - P[L-1]
        # bestOdd : maximum candidate among L odd  for dp[L-1] + P[L-1]
        # When L = 0 (even): candidate is dp[-1] - P[-1] = 0.
        bestEven = 0  
        bestOdd = -10**18  # a very small number
        
        # After index 0, possibility to start new segment at L = 1 (which is odd):
        bestOdd = max(bestOdd, dp[0] + P[0])
        
        # Process indices 1 .. n - 1
        for i in range(1, n):
            # update prefix alternating sum P[i]
            P[i] = P[i-1] + sign(i) * nums[i]
            # dp[i] can be computed using the formula:
            # dp[i] = max( P[i] + bestEven, -P[i] + bestOdd )
            cand1 = P[i] + bestEven
            cand2 = -P[i] + bestOdd
            dp[i] = max(cand1, cand2)
            
            # Now update the best candidate for starting a new subarray at index (i+1)
            if i + 1 < n:
                if (i + 1) % 2 == 0:
                    # For new subarray starting at an even index, candidate = dp[i] - P[i]
                    bestEven = max(bestEven, dp[i] - P[i])
                else:
                    # For new subarray starting at an odd index, candidate = dp[i] + P[i]
                    bestOdd = max(bestOdd, dp[i] + P[i])
                    
        return dp[n-1]