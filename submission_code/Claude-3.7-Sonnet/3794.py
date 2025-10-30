class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        
        # Initialize finish times (when wizard i finishes potion j)
        finish = [[0 for _ in range(m)] for _ in range(n)]
        
        # Calculate finish times for the first potion
        finish[0][0] = skill[0] * mana[0]
        for i in range(1, n):
            finish[i][0] = finish[i-1][0] + skill[i] * mana[0]
        
        # Calculate finish times for the rest of the potions
        for j in range(1, m):
            # Calculate the start time for potion j
            max_start = 0
            time_to_reach = 0
            
            # Consider each wizard and find the latest possible start time
            for i in range(n):
                if i > 0:
                    time_to_reach += skill[i-1] * mana[j]
                potential_start = finish[i][j-1] - time_to_reach
                max_start = max(max_start, potential_start)
            
            # Calculate finish times for each wizard for the current potion
            finish[0][j] = max_start + skill[0] * mana[j]
            for i in range(1, n):
                finish[i][j] = finish[i-1][j] + skill[i] * mana[j]
        
        return finish[n-1][m-1]