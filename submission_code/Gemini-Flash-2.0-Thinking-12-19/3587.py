class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        dp = [[0] * n for _ in range(k)]
        for city in range(n):
            dp[0][city] = stayScore[0][city]
        
        for day in range(1, k):
            for city in range(n):
                max_score = 0
                for prev_city in range(n):
                    # Travel from prev_city to city
                    travel_val = dp[day-1][prev_city] + travelScore[prev_city][city]
                    max_score = max(max_score, travel_val)
                # Stay in city
                stay_val = dp[day-1][city] + stayScore[day][city]
                max_score = max(max_score, stay_val)
                dp[day][city] = max_score
                
        max_total_score = 0
        for city in range(n):
            max_total_score = max(max_total_score, dp[k-1][city])
            
        return max_total_score