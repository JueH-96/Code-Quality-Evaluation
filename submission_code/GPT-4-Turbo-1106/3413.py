class Solution:
    def findWinningPlayer(self, skills: list[int], k: int) -> int:
        if k == 1:
            return skills.index(max(skills))
        
        wins = 0
        current_winner_index = 0
        current_winner_skill = skills[0]
        
        for i in range(1, len(skills)):
            if skills[i] > current_winner_skill:
                current_winner_skill = skills[i]
                current_winner_index = i
                wins = 1
            else:
                wins += 1
            
            if wins == k:
                return current_winner_index
        
        return current_winner_index