class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        # Start by assuming the champion candidate is 0
        champion_candidate = 0
        
        # If a team i is stronger than the current champion_candidate,
        # then team i becomes the new champion_candidate.
        for i in range(1, n):
            if grid[i][champion_candidate] == 1:
                champion_candidate = i
        
        # Confirm that the chosen champion_candidate is indeed the champion.
        # Since the problem guarantees consistency and exactly one champion,
        # this step will simply verify that there's no contradiction.
        for j in range(n):
            if j != champion_candidate and grid[j][champion_candidate] == 1:
                # If some team j is stronger than our candidate,
                # the input would be inconsistent with the guarantee.
                # But given the constraints, this won't happen.
                # We could return an error or handle it,
                # but we'll follow the specification and trust the input.
                return -1
        
        return champion_candidate