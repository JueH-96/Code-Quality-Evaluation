class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        if k >= n - 1:
            return skills.index(max(skills))
        
        winner = 0
        wins = 0
        
        for i in range(1, n):
            if skills[winner] > skills[i]:
                wins += 1
            else:
                winner = i
                wins = 1
            
            if wins == k:
                return winner
        
        return winner