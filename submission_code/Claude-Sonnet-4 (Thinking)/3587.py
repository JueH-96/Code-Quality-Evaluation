class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp[j] = maximum points ending in city j after current day
        dp = [0] * n
        
        # Day 0: we can start in any city
        for j in range(n):
            # Start in city j and stay
            dp[j] = stayScore[0][j]
            # Start in any city curr and travel to j
            for curr in range(n):
                if curr != j:
                    dp[j] = max(dp[j], travelScore[curr][j])
        
        # Days 1 to k-1
        for i in range(1, k):
            new_dp = [0] * n
            for j in range(n):
                # Option 1: stay in city j
                new_dp[j] = dp[j] + stayScore[i][j]
                # Option 2: travel from any city curr to j
                for curr in range(n):
                    if curr != j:
                        new_dp[j] = max(new_dp[j], dp[curr] + travelScore[curr][j])
            dp = new_dp
        
        return max(dp)