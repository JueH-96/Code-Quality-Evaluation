class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        max_val = max(skills[:k+1])
        return skills.index(max_val)