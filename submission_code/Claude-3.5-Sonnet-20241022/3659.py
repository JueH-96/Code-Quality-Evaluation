class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        
        # dp[i][j][xor] represents number of paths to (i,j) with XOR value xor
        dp = {}
        
        def dfs(i: int, j: int, curr_xor: int) -> int:
            # Base cases
            if i >= m or j >= n:
                return 0
            
            # If we reached destination, check if XOR equals k
            if i == m-1 and j == n-1:
                return 1 if curr_xor ^ grid[i][j] == k else 0
            
            # If we've seen this state before, return cached result
            state = (i, j, curr_xor)
            if state in dp:
                return dp[state]
            
            # Calculate new XOR value after including current cell
            new_xor = curr_xor ^ grid[i][j]
            
            # Try both possible moves (right and down) and sum their results
            paths = 0
            # Move right
            paths = (paths + dfs(i, j+1, new_xor)) % MOD
            # Move down
            paths = (paths + dfs(i+1, j, new_xor)) % MOD
            
            # Cache and return result
            dp[state] = paths
            return paths
        
        return dfs(0, 0, 0)