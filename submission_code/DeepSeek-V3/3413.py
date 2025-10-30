class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        max_skill = max(skills)
        max_index = skills.index(max_skill)
        
        if k >= n:
            return max_index
        
        current_winner = 0
        consecutive_wins = 0
        
        for i in range(1, n):
            if skills[current_winner] > skills[i]:
                consecutive_wins += 1
            else:
                current_winner = i
                consecutive_wins = 1
            
            if consecutive_wins == k:
                return current_winner
        
        return max_index