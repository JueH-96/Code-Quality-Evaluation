class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n, m = len(skill), len(mana)
        time = [[0] * m for _ in range(n)]
        
        # Calculate the time taken by each wizard for each potion
        for i in range(n):
            for j in range(m):
                time[i][j] = skill[i] * mana[j]
        
        # Initialize the minimum time required for each potion
        min_time = [float('inf')] * m
        
        # Iterate through the potions and calculate the minimum time required
        for j in range(m):
            for i in range(j, n):
                if i == j:
                    min_time[j] = time[i][j]
                else:
                    min_time[j] = min(min_time[j], min_time[j-1] + time[i][j])
        
        return min_time[-1]