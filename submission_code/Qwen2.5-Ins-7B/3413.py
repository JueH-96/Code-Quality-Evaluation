class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        max_skill = max(skills)
        max_index = skills.index(max_skill)
        wins = 0
        
        for i in range(n):
            if skills[i] > skills[(i + max_index) % n]:
                wins += 1
                if wins == k:
                    return max_index
            else:
                wins = 0
            max_index = (max_index + 1) % n
        return max_index