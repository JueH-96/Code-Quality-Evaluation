class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        
        # Time when each wizard finishes each potion
        finish_time = [[0] * m for _ in range(n)]
        
        # Initialize the first potion's finish times for all wizards
        for i in range(n):
            if i == 0:
                finish_time[i][0] = skill[i] * mana[0]
            else:
                finish_time[i][0] = finish_time[i-1][0] + skill[i] * mana[0]
        
        # Fill the finish times for all potions and all wizards
        for j in range(1, m):
            for i in range(n):
                if i == 0:
                    finish_time[i][j] = finish_time[i][j-1] + skill[i] * mana[j]
                else:
                    # Wizard i can only start after wizard i-1 has finished the current potion
                    # and after wizard i has finished the previous potion
                    start_time = max(finish_time[i-1][j], finish_time[i][j-1])
                    finish_time[i][j] = start_time + skill[i] * mana[j]
        
        # The last wizard's finish time for the last potion is the total time required
        return finish_time[n-1][m-1]