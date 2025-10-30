class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        result = []
        m, n = len(grid), len(grid[0])
        row, col = 0, 0
        direction = 1  # 1 for right, -1 for left

        while row < m and col < n:
            result.append(grid[row][col])
            row += 1
            col += direction

            if direction == 1 and col == n:
                row -= 1
                col = n - 1
                direction = -1
            elif direction == -1 and col < 0:
                row -= 1
                col = 0
                direction = 1

        return result