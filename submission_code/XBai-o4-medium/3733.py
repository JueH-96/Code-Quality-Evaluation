from typing import List

class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if n == 0:
            return 0
        m = len(grid[0])
        if m == 0:
            return 0
        
        # Directions: 0: (1,1), 1: (1,-1), 2: (-1,-1), 3: (-1,1)
        directions = [ (1, 1), (1, -1), (-1, -1), (-1, 1) ]
        
        # Precompute dp arrays: dp[dir][expected][i][j]
        # expected is 0 or 2, represented as 0 and 1 in the indices
        dp = [ [ [ [0]*m for _ in range(n) ] for _ in range(2) ] for _ in range(4) ]
        
        for dir_idx in range(4):
            dr, dc = directions[dir_idx]
            # Determine the iteration order for i and j
            if dir_idx == 0: # (1,1)
                irange = range(n-1, -1, -1)
                jrange = range(m-1, -1, -1)
            elif dir_idx == 1: # (1,-1)
                irange = range(n-1, -1, -1)
                jrange = range(m)
            elif dir_idx == 2: # (-1,-1)
                irange = range(n)
                jrange = range(m)
            elif dir_idx == 3: # (-1,1)
                irange = range(n)
                jrange = range(m-1, -1, -1)
            else:
                assert False, "Invalid dir_idx"
            
            for i in irange:
                for j in jrange:
                    # Compute dp for expected 0 and 2
                    next_i = i + dr
                    next_j = j + dc
                    # Expected 0
                    if 0 <= next_i < n and 0 <= next_j < m:
                        if grid[next_i][next_j] == 0:
                            dp[dir_idx][0][i][j] = 1 + dp[dir_idx][1][next_i][next_j]
                        else:
                            dp[dir_idx][0][i][j] = 0
                    else:
                        dp[dir_idx][0][i][j] = 0
                    # Expected 2
                    if 0 <= next_i < n and 0 <= next_j < m:
                        if grid[next_i][next_j] == 2:
                            dp[dir_idx][1][i][j] = 1 + dp[dir_idx][0][next_i][next_j]
                        else:
                            dp[dir_idx][1][i][j] = 0
                    else:
                        dp[dir_idx][1][i][j] = 0
        
        # Precompute forward_steps for each cell and direction
        forward_steps = [ [ [0]*m for _ in range(n) ] for _ in range(4) ]
        for dir_idx in range(4):
            dr, dc = directions[dir_idx]
            for i in range(n):
                for j in range(m):
                    if grid[i][j] != 1:
                        forward_steps[dir_idx][i][j] = 0
                    else:
                        next_i = i + dr
                        next_j = j + dc
                        if 0 <= next_i < n and 0 <= next_j < m:
                            if grid[next_i][next_j] == 2:
                                steps_after = dp[dir_idx][0][next_i][next_j]
                                forward_steps[dir_idx][i][j] = 1 + 1 + steps_after
                            else:
                                forward_steps[dir_idx][i][j] = 1
                        else:
                            forward_steps[dir_idx][i][j] = 1
        
        max_length = 0
        
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    for dir_idx in range(4):
                        m_val = forward_steps[dir_idx][i][j]
                        if m_val == 0:
                            continue
                        dr, dc = directions[dir_idx]
                        for k in range(m_val):
                            x = i + dr * k
                            y = j + dc * k
                            s = k + 1
                            expected_val = 2 if s % 2 == 1 else 0
                            # Map expected_val to index (0 for 0, 1 for 2)
                            expected_idx = 1 if expected_val == 2 else 0
                            dir2 = (dir_idx + 1) % 4
                            # Get steps_backward
                            steps_backward = dp[dir2][expected_idx][x][y]
                            total_length = (k + 1) + steps_backward
                            if total_length > max_length:
                                max_length = total_length
                        # Also consider the forward part without turning
                        if m_val > max_length:
                            max_length = m_val
        
        return max_length