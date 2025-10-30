class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def is_square(lst):
            return len(set(lst)) == 1

        m, n = len(grid), len(grid[0])
        if any(len(row) != n for row in grid):
            return False
        if any(len(row) < m for row in grid):
            return False
        if not is_square(map(sum, grid)):
            return False
        if not is_square([sum(col) for col in grid]):
            return False
        return True