class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        
        # Initialize the time taken for each potion
        potion_time = [0] * m
        
        # Iterate through each potion
        for j in range(m):
            # Calculate the time taken for each wizard for the current potion
            for i in range(n):
                if j == 0:
                    # For the first potion, it starts at time 0
                    potion_time[j] += skill[i] * mana[j]
                else:
                    # For subsequent potions, it starts after the previous potion is done
                    potion_time[j] = max(potion_time[j], potion_time[j - 1]) + skill[i] * mana[j]
        
        # The total time is the time taken for the last potion to be completed
        return potion_time[-1]