class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        if k >= n:
            max_skill = max(skills)
            return skills.index(max_skill)
        
        current = 0
        wins = 0
        
        for i in range(1, n):
            if skills[current] > skills[i]:
                wins += 1
            else:
                current = i
                wins = 1
            if wins == k:
                return current
        
        return current