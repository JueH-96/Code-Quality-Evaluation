class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp[day][city] = maximum score achievable by being in city at end of day
        dp = [[-1] * n for _ in range(k)]
        
        # Day 0: Tourist can start in any city
        for start_city in range(n):
            # Option 1: Start and stay in start_city
            dp[0][start_city] = stayScore[0][start_city]
            
            # Option 2: Start in start_city and travel to another city
            for dest_city in range(n):
                if dest_city != start_city:
                    dp[0][dest_city] = max(dp[0][dest_city], travelScore[start_city][dest_city])
        
        # Fill dp for remaining days
        for day in range(1, k):
            for curr_city in range(n):
                # Option 1: Stay in current city
                if dp[day-1][curr_city] != -1:
                    dp[day][curr_city] = max(dp[day][curr_city], 
                                            dp[day-1][curr_city] + stayScore[day][curr_city])
                
                # Option 2: Travel from another city
                for prev_city in range(n):
                    if prev_city != curr_city and dp[day-1][prev_city] != -1:
                        dp[day][curr_city] = max(dp[day][curr_city], 
                                                dp[day-1][prev_city] + travelScore[prev_city][curr_city])
        
        # Return maximum score across all cities on the last day
        return max(dp[k-1])