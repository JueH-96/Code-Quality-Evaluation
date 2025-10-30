class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        from collections import deque

        # Helper function to perform DP for each child
        def dp_child(start_i, start_j, moves, blocked):
            dp = [[-1] * n for _ in range(n)]
            dp[start_i][start_j] = fruits[start_i][start_j] if (start_i, start_j) not in blocked else -1
            for i in range(start_i, n):
                for j in range(n):
                    if dp[i][j] == -1:
                        continue
                    for di, dj in moves:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in blocked:
                            val = dp[i][j] + fruits[ni][nj]
                            if val > dp[ni][nj]:
                                dp[ni][nj] = val
            # Reconstruct the path
            path = set()
            i, j = n - 1, n - 1
            if dp[i][j] == -1:
                return 0, path  # No path found
            while (i, j) != (start_i, start_j):
                path.add((i, j))
                max_val = -1
                prev_i, prev_j = -1, -1
                for di, dj in [(-d[0], -d[1]) for d in moves]:
                    pi, pj = i + di, j + dj
                    if 0 <= pi < n and 0 <= pj < n and dp[pi][pj] != -1:
                        if dp[pi][pj] > max_val:
                            max_val = dp[pi][pj]
                            prev_i, prev_j = pi, pj
                i, j = prev_i, prev_j
            path.add((start_i, start_j))
            return dp[n - 1][n - 1], path

        total = 0
        blocked = set()

        # Child 1: Starts at (0, 0), moves: (1, 0), (0, 1), (1, 1)
        moves1 = [(1, 0), (0, 1), (1, 1)]
        val1, path1 = dp_child(0, 0, moves1, blocked)
        total += val1
        blocked |= path1

        # Child 2: Starts at (0, n -1), moves: (1, -1), (1, 0), (1, 1)
        moves2 = [(1, -1), (1, 0), (1, 1)]
        val2, path2 = dp_child(0, n - 1, moves2, blocked)
        total += val2
        blocked |= path2

        # Child 3: Starts at (n -1, 0), moves: (-1, 1), (0, 1), (1, 1)
        moves3 = [(-1, 1), (0, 1), (1, 1)]
        val3, path3 = dp_child(n - 1, 0, moves3, blocked)
        total += val3

        return total