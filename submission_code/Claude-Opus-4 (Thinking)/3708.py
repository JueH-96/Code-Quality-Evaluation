class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        m, n = len(grid), len(grid[0])
        full_path = []
        
        for i in range(m):
            if i % 2 == 0:  # Even rows: left to right
                full_path.extend(grid[i])
            else:  # Odd rows: right to left
                full_path.extend(grid[i][::-1])
        
        # Skip every alternate cell (take every 2nd element starting from index 0)
        return full_path[::2]