class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        result = []
        m = len(grid)
        if m == 0:
            return result
        n = len(grid[0])
        for i in range(m):
            if i % 2 == 0:
                # Left to right, skip every other starting at 0 (0, 2, 4...)
                for j in range(0, n, 2):
                    result.append(grid[i][j])
            else:
                # Right to left, skip every other starting at 0 (or adjusted for odd/even)
                for j in range(n - 1, -1, -1):
                    # The positions to take are n-1, n-3, etc., which is equivalent to (n-1 - even positions)
                    # So j starts from n-1, and we take j if (n-1 - j) is even, i.e., j is n-1, n-3, etc.
                    if (n - 1 - j) % 2 == 0:
                        result.append(grid[i][j])
        return result