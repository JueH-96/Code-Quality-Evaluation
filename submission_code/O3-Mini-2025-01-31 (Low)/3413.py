from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        # If k is 0, the game hasn't started yet, but since k is positive, no need to handle.
        # Find the maximum skilled player's index.
        max_skill = max(skills)
        max_index = skills.index(max_skill)
        
        # If k is large enough, the maximum skilled player will eventually win regardless.
        # Specifically, if k >= len(skills) then even if simulation takes several rounds,
        # the best player will become the champion.
        # However, we can simulate up to when a player wins k consecutive games.
        
        current_champion = skills[0]
        champion_index = 0  # initial index of current champion in the original input
        wins_in_a_row = 0
        
        # If the best player is already at the beginning or if k==0 situation,
        # but since k is always >= 1, we simulate.
        # When current_champion is the maximum skilled player, it will never lose.
        # So, if k is huge, we can return it immediately.
        if current_champion == max_skill:
            # If current champion is max, then it will eventually accumulate k wins.
            return champion_index
        
        # Otherwise, simulate the games
        # Since there are n players, the worst-case simulation will involve at most (n-1) challengers
        # before the maximum skilled player becomes current champion and never loses.
        # But we cannot simply simulate infinitely because k can be huge;
        # however, once the maximum player is the champion, they will be champion regardless.
        for challenger in skills[1:]:
            # Play a game between current_champion and challenger:
            if current_champion > challenger:
                # current champion wins, increment count and push challenger to end implicitly.
                wins_in_a_row += 1
            else:
                # challenger wins, becomes the new champion with win streak reset to 1.
                current_champion = challenger
                champion_index = skills.index(challenger)  # get original index
                wins_in_a_row = 1
                
            # If current champion has reached k consecutive wins, that's our winner.
            if wins_in_a_row >= k:
                return champion_index
            
            # If current champion is the best skilled player, they will always win afterwards.
            if current_champion == max_skill:
                return champion_index
        
        # If simulation did not return yet (k > number of simulated games),
        # then the best skilled player is ultimately the winner.
        return max_index