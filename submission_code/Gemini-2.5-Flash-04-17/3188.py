from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        # The problem statement guarantees that for any two distinct teams i and j,
        # either grid[i][j] == 1 (team i is stronger) or grid[j][i] == 1 (team j is stronger),
        # but not both. It also guarantees transitivity (if a beats b and b beats c, then a beats c).
        # These properties together imply a strict total ordering of teams by strength.
        
        # A team 'a' is the champion if there is no team 'b' that is stronger than 'a'.
        # In a total ordering, this definition applies uniquely to the strongest team.
        # The strongest team is the one that is stronger than every other team.
        # We can find this strongest team efficiently.

        # We use a single pass approach to find the strongest team.
        # We maintain a 'champion_candidate' which is the strongest team encountered so far.
        
        # Initialize the candidate to the first team.
        champion_candidate = 0
        
        # Iterate through the rest of the teams (from the second team onwards).
        for i in range(1, n):
            # Compare the current 'champion_candidate' with team 'i'.
            # grid[a][b] == 1 means team a is stronger than team b.
            
            # If grid[champion_candidate][i] == 0, it means the current champion_candidate
            # is NOT stronger than team i. Due to the problem constraints, this implies
            # team i IS stronger than the champion_candidate (grid[i][champion_candidate] == 1).
            # If team i is stronger than our current candidate, then team i becomes
            # the new potential champion, as the old candidate cannot be the true champion.
            if grid[champion_candidate][i] == 0:
                champion_candidate = i
            # else (if grid[champion_candidate][i] == 1):
                # The current champion_candidate is stronger than team i.
                # Team i cannot be the overall strongest team. The current
                # champion_candidate remains the strongest team found among
                # the teams examined so far.
        
        # After checking against all other teams (from 1 to n-1), the final
        # 'champion_candidate' is the team that was never found to be weaker
        # than any team it was compared against in the process. By the property
        # of total ordering, this must be the strongest team overall, which is the champion.
        return champion_candidate