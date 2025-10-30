class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        prev = [0] * n
        for day in range(k):
            curr = [0] * n
            for j in range(n):
                stay = prev[j] + stayScore[day][j]
                travel_max = max(prev[m] + travelScore[m][j] for m in range(n))
                curr[j] = max(stay, travel_max)
            prev = curr
        return max(prev)