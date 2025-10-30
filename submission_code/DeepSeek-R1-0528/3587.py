class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        dp_prev = [0] * n
        for j in range(n):
            max_travel = 0
            for c in range(n):
                if travelScore[c][j] > max_travel:
                    max_travel = travelScore[c][j]
            dp_prev[j] = max(stayScore[0][j], max_travel)
        
        if k == 1:
            return max(dp_prev)
        
        for i in range(1, k):
            travel_option = [-10**9] * n
            for c in range(n):
                for j in range(n):
                    total = dp_prev[c] + travelScore[c][j]
                    if total > travel_option[j]:
                        travel_option[j] = total
            dp_curr = [0] * n
            for j in range(n):
                stay_option = dp_prev[j] + stayScore[i][j]
                dp_curr[j] = max(stay_option, travel_option[j])
            dp_prev = dp_curr
        
        return max(dp_prev)