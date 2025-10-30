class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        dirs = [(1,1), (1,-1), (-1,-1), (-1,1)]
        
        dp = [[[[0]*3 for _ in range(m)] for __ in range(n)] for ___ in range(4)]
        
        for d in range(4):
            dr, dc = dirs[d]
            i_range = range(n-1, -1, -1) if dr == 1 else range(n)
            j_range = range(m-1, -1, -1) if dc == 1 else range(m)
            for i in i_range:
                for j in j_range:
                    if grid[i][j] == 1:
                        ni, nj = i + dr, j + dc
                        if ni < 0 or ni >= n or nj < 0 or nj >= m:
                            dp[d][i][j][0] = 1
                        else:
                            dp[d][i][j][0] = 1 + dp[d][ni][nj][1]
                    else:
                        dp[d][i][j][0] = 0
                    
                    if grid[i][j] == 2:
                        ni, nj = i + dr, j + dc
                        if ni < 0 or ni >= n or nj < 0 or nj >= m:
                            dp[d][i][j][1] = 1
                        else:
                            dp[d][i][j][1] = 1 + dp[d][ni][nj][2]
                    else:
                        dp[d][i][j][1] = 0
                    
                    if grid[i][j] == 0:
                        ni, nj = i + dr, j + dc
                        if ni < 0 or ni >= n or nj < 0 or nj >= m:
                            dp[d][i][j][2] = 1
                        else:
                            dp[d][i][j][2] = 1 + dp[d][ni][nj][1]
                    else:
                        dp[d][i][j][2] = 0
        
        ans = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    for d0 in range(4):
                        L0 = dp[d0][i][j][0]
                        if L0 == 0:
                            continue
                        dr0, dc0 = dirs[d0]
                        d1 = (d0 + 1) % 4
                        dr1, dc1 = dirs[d1]
                        for t in range(L0):
                            i_t = i + t * dr0
                            j_t = j + t * dc0
                            if t == 0:
                                state_here = 0
                            elif t % 2 == 1:
                                state_here = 1
                            else:
                                state_here = 2
                            
                            i_next = i_t + dr1
                            j_next = j_t + dc1
                            if i_next < 0 or i_next >= n or j_next < 0 or j_next >= m:
                                candidate = t + 1
                            else:
                                if state_here == 0:
                                    state_new = 1
                                elif state_here == 1:
                                    state_new = 2
                                else:
                                    state_new = 1
                                l_new = dp[d1][i_next][j_next][state_new]
                                candidate = t + 1 + l_new
                            
                            if candidate > ans:
                                ans = candidate
        return ans