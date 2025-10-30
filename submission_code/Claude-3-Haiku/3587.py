class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        dp = [[[0] * (k + 1) for _ in range(n)] for _ in range(k + 1)]
        
        for day in range(k):
            for city in range(n):
                # Stay in the current city
                dp[day + 1][city][day + 1] = max(dp[day + 1][city][day + 1], dp[day][city][day] + stayScore[day][city])
                
                # Move to another city
                for dest in range(n):
                    if dest != city:
                        dp[day + 1][dest][day + 1] = max(dp[day + 1][dest][day + 1], dp[day][city][day] + travelScore[city][dest])
        
        return max(dp[k][i][k] for i in range(n))