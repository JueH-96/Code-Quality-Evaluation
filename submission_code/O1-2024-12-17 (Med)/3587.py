class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp[i][c] will store the maximum points achievable at the end of day i
        # if the tourist is in city c.
        dp = [[0]*n for _ in range(k)]
        
        # Initialize for day 0: starting in city c earns stayScore[0][c].
        for c in range(n):
            dp[0][c] = stayScore[0][c]
        
        # Fill dp for each day
        for i in range(1, k):
            # Precompute the best possible "travel" transition into each city c
            bestTravelTo = [0]*n
            for c in range(n):
                bestVal = float('-inf')
                for prevCity in range(n):
                    if prevCity != c:  # move must be from a different city
                        bestVal = max(bestVal, dp[i-1][prevCity] + travelScore[prevCity][c])
                bestTravelTo[c] = bestVal
            
            # Now compute dp[i][c] considering both staying and traveling
            for c in range(n):
                stayPoints = dp[i-1][c] + stayScore[i][c]
                movePoints = bestTravelTo[c]
                dp[i][c] = max(stayPoints, movePoints)
        
        # The answer is the maximum across all cities on the last day
        return max(dp[k-1])