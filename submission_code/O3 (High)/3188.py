from typing import List

class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        """
        The champion is the unique team that is not beaten by any other team.
        Under the given constraints this is equivalent to the row that contains
        a '1' against every other team (because if team *i* does not beat team *j*,
        then team *j* beats team *i*).
        """
        n = len(grid)

        for i in range(n):
            # A candidate is champion if it defeats everybody else
            # i.e. its entire row (except the diagonal) is filled with 1's.
            if all(grid[i][j] == 1 or i == j for j in range(n)):
                return i

        # According to problem constraints, a champion always exists.
        return -1