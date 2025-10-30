class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        prev = stayScore[0][:]
        
        if k == 1:
            return max(prev)
        
        for i in range(1, k):
            M = [-10**9] * n
            for p in range(n):
                row = travelScore[p]
                for j in range(n):
                    if j == p:
                        continue
                    total = prev[p] + row[j]
                    if total > M[j]:
                        M[j] = total
            
            curr = [0] * n
            for j in range(n):
                stay_opt = prev[j] + stayScore[i][j]
                curr[j] = max(stay_opt, M[j])
            
            prev = curr
        
        return max(prev)