from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        prev_dp = [stayScore[0][j] for j in range(n)]
        
        for i in range(1, k):
            curr_dp = [0] * n
            for j in range(n):
                stay = prev_dp[j] + stayScore[i][j]
                move = 0
                for l in range(n):
                    if l != j:
                        temp = prev_dp[l] + travelScore[l][j]
                        if temp > move:
                            move = temp
                curr_dp[j] = max(stay, move)
            prev_dp = curr_dp[:]
        
        return max(prev_dp)