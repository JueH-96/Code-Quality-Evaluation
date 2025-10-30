class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        King = 0
        wins = 0
        for i in range(1, n):
            if skills[King] > skills[i]:
                wins += 1
            else:
                King = i
                wins = 1
            if wins == k:
                return King
        return King