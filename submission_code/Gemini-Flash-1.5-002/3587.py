class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        dp = {}

        def solve(day, curr_city):
            if day == k:
                return 0

            if (day, curr_city) in dp:
                return dp[(day, curr_city)]

            max_points = stayScore[day][curr_city] + solve(day + 1, curr_city)

            for next_city in range(n):
                if next_city != curr_city:
                    max_points = max(max_points, travelScore[curr_city][next_city] + solve(day + 1, next_city))

            dp[(day, curr_city)] = max_points
            return max_points

        max_overall_points = 0
        for start_city in range(n):
            max_overall_points = max(max_overall_points, solve(0, start_city))

        return max_overall_points