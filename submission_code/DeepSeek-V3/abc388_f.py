class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        
        if m == 0:
            return 0
        
        # Initialize a DP table where dp[i][j] is the finish time of j-th potion by i-th wizard
        # To optimize space, we can use two 1D arrays: previous and current
        prev_finish = [0] * n
        for j in range(m):
            current_finish = [0] * n
            for i in range(n):
                if j == 0:
                    if i == 0:
                        start_time = 0
                    else:
                        start_time = prev_finish[i-1]
                else:
                    if i == 0:
                        start_time = prev_finish[i]
                    else:
                        start_time = max(prev_finish[i], current_finish[i-1])
                current_finish[i] = start_time + skill[i] * mana[j]
            prev_finish = current_finish
        
        return prev_finish[-1]