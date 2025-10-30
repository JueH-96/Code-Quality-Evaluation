import collections # Not used in the final solution, but often part of standard imports
from typing import List # Required for type hints

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        """
        Finds the champion team in a tournament based on a strength grid.

        The problem defines the champion as a team 'a' for which there is no 
        team 'b' that is stronger than 'a'.
        In terms of the input grid, `grid[b][a] == 1` means team 'b' is stronger 
        than team 'a'.
        Therefore, team 'a' is the champion if `grid[b][a] == 0` for all `b` 
        where `b != a`.
        Since the problem states `grid[a][a] == 0`, the condition simplifies to:
        Team 'a' is the champion if `grid[b][a] == 0` for all `b` from 0 to n-1.
        This means the champion team corresponds to the column in the grid that 
        contains only zeros.

        This algorithm iterates through each team, considering it as a potential
        champion. For each potential champion `candidate_champion`, it checks 
        if any other team (`opponent`) is stronger by examining the corresponding 
        column `grid[...][candidate_champion]`. If a '1' is found in the column, 
        it means the candidate is beaten by someone, so it cannot be the champion.
        If the entire column is checked and no '1's are found (meaning no opponent 
        is stronger), then the `candidate_champion` is the true champion.

        Args:
            grid: A 0-indexed 2D list of integers (0 or 1) of size n * n. 
                  `grid[i][j] == 1` indicates team `i` is stronger than team `j`.
                  `grid[i][j] == 0` indicates team `j` is stronger than team `i`.
                  Constraints guarantee `n >= 2`, `grid[i][i] == 0`, and 
                  `grid[i][j] != grid[j][i]` for `i != j`. Transitivity also holds.

        Returns:
            The index (0 to n-1) of the champion team. The problem guarantees 
            that exactly one champion exists.
        """
        n = len(grid) # Get the number of teams (size of the grid)
        
        # Iterate through each team index, considering it as a potential champion.
        for candidate_champion in range(n):
            # Assume the current candidate is the champion until proven otherwise.
            # 'is_beaten' flag tracks if we find any team stronger than the candidate.
            is_beaten = False 
            
            # Check if any team 'opponent' is stronger than the 'candidate_champion'.
            # This involves checking the column corresponding to 'candidate_champion'.
            for opponent in range(n):
                # If grid[opponent][candidate_champion] is 1, it means 'opponent' 
                # is stronger than 'candidate_champion'.
                # Note: We don't need to explicitly check `opponent != candidate_champion`
                # because the constraint `grid[i][i] == 0` ensures that 
                # `grid[candidate_champion][candidate_champion]` will be 0, 
                # so the condition `grid[opponent][candidate_champion] == 1` will 
                # naturally be false when opponent == candidate_champion.
                if grid[opponent][candidate_champion] == 1:
                    # The candidate team has been beaten by the opponent team.
                    is_beaten = True 
                    # Since the candidate is beaten, it cannot be the champion.
                    # We can stop checking other opponents for this candidate and
                    # move to the next potential champion.
                    break 
            
            # After checking all potential opponents, if the 'is_beaten' flag is still False,
            # it means no team was found to be stronger than the 'candidate_champion'.
            # Therefore, this candidate is the champion.
            if not is_beaten:
                return candidate_champion 
        
        # According to the problem description, a unique champion is guaranteed to exist.
        # Therefore, the loop above will always find and return the champion.
        # This line is theoretically unreachable under the given constraints.
        # It's included for completeness, but returning -1 or raising an error
        # might be appropriate if the guarantee wasn't absolute.
        return -1 # Should not be reached given the problem constraints.