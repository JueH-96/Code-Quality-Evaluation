class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        max_val = max(skills)
        highest_index = skills.index(max_val)  # Find the index of the highest skill because that's the only one that can win k games in a row
        
        # Check if the max skill is at the start, otherwise, it will not win k games in a row because it cannot win the first round
        if highest_index != 0:
            return -1
        
        # Count how many times the player with the highest skill could win in a row
        win_streak = (len(skills) - 1) // 2 + 1
        if win_streak >= k:
            return highest_index
        else:
            return -1