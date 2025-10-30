class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Maximum possible sum is bounded
        max_val = max(rewardValues)
        max_possible = max_val * 2  # Upper bound on achievable sum
        
        # dp[i] = True if sum i is achievable
        dp = [False] * (max_possible + 1)
        dp[0] = True
        
        # Process each value
        for value in rewardValues:
            # Create new dp array to avoid using same value multiple times
            new_dp = dp[:]
            
            # Check all possible sums we can extend
            for current_sum in range(max_possible + 1):
                if dp[current_sum] and value > current_sum:
                    new_sum = current_sum + value
                    if new_sum <= max_possible:
                        new_dp[new_sum] = True
            
            dp = new_dp
        
        # Find maximum achievable sum
        for i in range(max_possible, -1, -1):
            if dp[i]:
                return i
        
        return 0