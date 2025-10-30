class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        dp = {}

        def solve(day, city):
            if day == k:
                return 0
            if (day, city) in dp:
                return dp[(day, city)]

            max_score = 0
            # Stay in the current city
            max_score = max(max_score, stayScore[day][city] + solve(day + 1, city))

            # Travel to another city
            for next_city in range(n):
                if next_city != city:
                    max_score = max(max_score, travelScore[city][next_city] + solve(day + 1, next_city))
            
            dp[(day, city)] = max_score
            return max_score

        max_total_score = 0
        for start_city in range(n):
            max_total_score = max(max_total_score, solve(0, start_city))

        return max_total_score