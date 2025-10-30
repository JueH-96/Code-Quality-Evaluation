class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        # Precompute new_dir_indices for each direction
        new_dir_indices = [0] * 4
        for dir_idx in range(4):
            dx, dy = directions[dir_idx]
            new_dx = dy
            new_dy = -dx
            for nd in range(4):
                if directions[nd] == (new_dx, new_dy):
                    new_dir_indices[dir_idx] = nd
                    break
        
        # Initialize max_0 and max_2 arrays for each direction
        max_0 = [[[0 for _ in range(m)] for __ in range(n)] for ___ in range(4)]
        max_2 = [[[0 for _ in range(m)] for __ in range(n)] for ___ in range(4)]
        
        for dir_idx in range(4):
            dx, dy = directions[dir_idx]
            if dir_idx == 0:  # (1,1) direction: process from bottom-right to top-left
                for i in range(n-1, -1, -1):
                    for j in range(m-1, -1, -1):
                        next_i = i + dx
                        next_j = j + dy
                        if next_i < n and next_j < m:
                            if grid[next_i][next_j] == 0:
                                max_0[dir_idx][i][j] = 1 + max_2[dir_idx][next_i][next_j]
                            else:
                                max_0[dir_idx][i][j] = 0
                            if grid[next_i][next_j] == 2:
                                max_2[dir_idx][i][j] = 1 + max_0[dir_idx][next_i][next_j]
                            else:
                                max_2[dir_idx][i][j] = 0
                        else:
                            max_0[dir_idx][i][j] = 0
                            max_2[dir_idx][i][j] = 0
            elif dir_idx == 1:  # (1,-1) direction: process from bottom-left to top-right
                for i in range(n-1, -1, -1):
                    for j in range(m):
                        next_i = i + dx
                        next_j = j + dy
                        if 0 <= next_i < n and 0 <= next_j < m:
                            if grid[next_i][next_j] == 0:
                                max_0[dir_idx][i][j] = 1 + max_2[dir_idx][next_i][next_j]
                            else:
                                max_0[dir_idx][i][j] = 0
                            if grid[next_i][next_j] == 2:
                                max_2[dir_idx][i][j] = 1 + max_0[dir_idx][next_i][next_j]
                            else:
                                max_2[dir_idx][i][j] = 0
                        else:
                            max_0[dir_idx][i][j] = 0
                            max_2[dir_idx][i][j] = 0
            elif dir_idx == 2:  # (-1,1) direction: process from top-right to bottom-left
                for i in range(n):
                    for j in range(m-1, -1, -1):
                        next_i = i + dx
                        next_j = j + dy
                        if 0 <= next_i < n and 0 <= next_j < m:
                            if grid[next_i][next_j] == 0:
                                max_0[dir_idx][i][j] = 1 + max_2[dir_idx][next_i][next_j]
                            else:
                                max_0[dir_idx][i][j] = 0
                            if grid[next_i][next_j] == 2:
                                max_2[dir_idx][i][j] = 1 + max_0[dir_idx][next_i][next_j]
                            else:
                                max_2[dir_idx][i][j] = 0
                        else:
                            max_0[dir_idx][i][j] = 0
                            max_2[dir_idx][i][j] = 0
            elif dir_idx == 3:  # (-1,-1) direction: process from top-left to bottom-right
                for i in range(n):
                    for j in range(m):
                        next_i = i + dx
                        next_j = j + dy
                        if 0 <= next_i < n and 0 <= next_j < m:
                            if grid[next_i][next_j] == 0:
                                max_0[dir_idx][i][j] = 1 + max_2[dir_idx][next_i][next_j]
                            else:
                                max_0[dir_idx][i][j] = 0
                            if grid[next_i][next_j] == 2:
                                max_2[dir_idx][i][j] = 1 + max_0[dir_idx][next_i][next_j]
                            else:
                                max_2[dir_idx][i][j] = 0
                        else:
                            max_0[dir_idx][i][j] = 0
                            max_2[dir_idx][i][j] = 0
        
        max_length = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] != 1:
                    continue
                for dir_idx in range(4):
                    dx1, dy1 = directions[dir_idx]
                    current_max = max_2[dir_idx][i][j] + 1
                    if current_max > max_length:
                        max_length = current_max
                    max_steps_initial = max_2[dir_idx][i][j]
                    for k in range(0, max_steps_initial + 1):
                        current_x = i + dx1 * k
                        current_y = j + dy1 * k
                        new_dir = new_dir_indices[dir_idx]
                        v = 2 if (k + 1) % 2 == 1 else 0
                        if v == 0:
                            max_new = max_0[new_dir][current_x][current_y]
                        else:
                            max_new = max_2[new_dir][current_x][current_y]
                        total = (k + 1) + max_new
                        if total > max_length:
                            max_length = total
        return max_length