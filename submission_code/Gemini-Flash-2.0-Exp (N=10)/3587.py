class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        dp = [[0] * n for _ in range(k + 1)]
        
        for day in range(k):
            for curr_city in range(n):
                # Option 1: Stay in the current city
                dp[day + 1][curr_city] = max(dp[day + 1][curr_city], dp[day][curr_city] + stayScore[day][curr_city])
                
                # Option 2: Travel to another city
                for next_city in range(n):
                    if curr_city != next_city:
                        dp[day + 1][next_city] = max(dp[day + 1][next_city], dp[day][curr_city] + travelScore[curr_city][next_city])

        return max(dp[k])