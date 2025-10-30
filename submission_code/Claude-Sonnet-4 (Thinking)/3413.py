class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        max_skill = max(skills)
        champion = 0  # Index of current champion
        win_streak = 0
        
        for challenger in range(1, len(skills)):
            if skills[champion] > skills[challenger]:
                # Champion wins, increment their streak
                win_streak += 1
            else:
                # Challenger wins, becomes new champion
                champion = challenger
                win_streak = 1
            
            # Check if we have a winner
            if win_streak == k or skills[champion] == max_skill:
                return champion
        
        # If we finish the loop, current champion has max skill
        return champion