class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        
        # Compute prefix sums of skills
        S = [0] * (n + 1)
        for i in range(n):
            S[i + 1] = S[i] + skill[i]
        
        # start[j] = start time of wizard 0 on potion j
        start = [0] * m
        
        for j in range(1, m):
            # Find the maximum constraint
            max_constraint = 0
            for i in range(n):
                # Wizard i must finish potion j-1 before starting potion j
                # finish[i][j-1] = start[j-1] + mana[j-1] * S[i+1]
                # start[i][j] = start[j] + mana[j] * S[i]
                # We need: start[j] + mana[j] * S[i] >= start[j-1] + mana[j-1] * S[i+1]
                constraint = start[j - 1] + mana[j - 1] * S[i + 1] - mana[j] * S[i]
                max_constraint = max(max_constraint, constraint)
            start[j] = max_constraint
        
        # The total time is when the last wizard finishes the last potion
        # finish[n-1][m-1] = start[m-1] + mana[m-1] * S[n]
        return start[m - 1] + mana[m - 1] * S[n]