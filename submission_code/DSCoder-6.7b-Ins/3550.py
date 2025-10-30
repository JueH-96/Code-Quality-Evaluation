class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        directions = [(0, 1), (1, 0)]
        max_sum = -float('inf')

        def dfs(i, j, rooks, cur_sum):
            nonlocal max_sum
            if rooks == 3:
                max_sum = max(max_sum, cur_sum)
                return
            for dx, dy in directions:
                x, y = i + dx, j + dy
                if 0 <= x < m and 0 <= y < n and (x, y) not in rooks:
                    rooks.add((x, y))
                    dfs(x, y, rooks, cur_sum + board[x][y])
                    rooks.remove((x, y))

        for i in range(m):
            for j in range(n):
                dfs(i, j, set([(i, j)]), board[i][j])

        return max_sum