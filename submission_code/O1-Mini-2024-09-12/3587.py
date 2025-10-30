class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        prev_dp = stayScore[0].copy()
        
        for day in range(1, k):
            curr_dp = [0] * n
            # Precompute max values for travel
            # For each destination city, find the max prev_dp[p] + travelScore[p][c]
            max_travel = [float('-inf')] * n
            for p in range(n):
                for c in range(n):
                    if p != c:
                        max_travel[c] = max(max_travel[c], prev_dp[p] + travelScore[p][c])
            for c in range(n):
                stay = prev_dp[c] + stayScore[day][c]
                travel = max_travel[c]
                curr_dp[c] = max(stay, travel)
            prev_dp = curr_dp
        return max(prev_dp)