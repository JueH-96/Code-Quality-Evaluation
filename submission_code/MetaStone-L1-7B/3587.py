class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        if k == 0:
            return 0
        
        # Initialize dp_prev for day 0
        dp_prev = [0] * n
        for j in range(n):
            option1 = stayScore[0][j]
            max_option2 = -float('inf')
            for m in range(n):
                temp = dp_prev[m] + travelScore[m][j]
                if temp > max_option2:
                    max_option2 = temp
            dp_prev[j] = max(option1, max_option2)
        
        for i in range(1, k):
            dp_current = [0] * n
            for j in range(n):
                option1 = dp_prev[j] + stayScore[i][j]
                max_option2 = -float('inf')
                for m in range(n):
                    temp = dp_prev[m] + travelScore[m][j]
                    if temp > max_option2:
                        max_option2 = temp
                dp_current[j] = max(option1, max_option2)
            dp_prev = dp_current.copy()
        
        return max(dp_prev)