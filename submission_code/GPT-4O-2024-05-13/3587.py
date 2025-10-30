class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize the dp array where dp[day][city] represents the maximum score achievable on 'day' ending in 'city'
        dp = [[-float('inf')] * n for _ in range(k)]
        
        # Base case: On day 0, the tourist can start in any city
        for city in range(n):
            dp[0][city] = stayScore[0][city]
        
        # Fill the dp array for each day
        for day in range(1, k):
            for curr_city in range(n):
                # Calculate the maximum score for staying in the current city
                dp[day][curr_city] = max(dp[day][curr_city], dp[day-1][curr_city] + stayScore[day][curr_city])
                
                # Calculate the maximum score for traveling to another city
                for prev_city in range(n):
                    if prev_city != curr_city:
                        dp[day][curr_city] = max(dp[day][curr_city], dp[day-1][prev_city] + travelScore[prev_city][curr_city] + stayScore[day][curr_city])
        
        # The answer is the maximum score achievable on the last day in any city
        return max(dp[k-1])

# Example usage:
# sol = Solution()
# print(sol.maxScore(2, 1, [[2,3]], [[0,2],[1,0]]))  # Output: 3
# print(sol.maxScore(3, 2, [[3,4,2],[2,1,2]], [[0,2,1],[2,0,4],[3,2,0]]))  # Output: 8