class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        current_champion = 0
        streak = 0
        n = len(skills)
        for i in range(1, n):
            if skills[i] > skills[current_champion]:
                current_champion = i
                streak = 1
            else:
                streak += 1
            if streak >= k:
                return current_champion
        return current_champion