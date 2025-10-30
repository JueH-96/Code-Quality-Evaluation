class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        
        # Child 1 must follow the diagonal path from (0,0) to (n-1,n-1)
        child1_fruits = sum(fruits[i][i] for i in range(n))
        
        # For child 2 starting at (0, n-1), we need to find the best path
        # Child 2 can move: down-left, down, down-right
        # Valid moves from (i,j): (i+1,j-1), (i+1,j), (i+1,j+1)
        
        # DP for child 2 - we only need to consider the upper-right triangle
        dp2 = [[-1] * n for _ in range(n)]
        dp2[0][n-1] = fruits[0][n-1]
        
        for i in range(n-1):
            for j in range(n):
                if dp2[i][j] == -1:
                    continue
                # Try all three moves
                for dj in [-1, 0, 1]:
                    ni, nj = i + 1, j + dj
                    if 0 <= nj < n and ni < n:
                        # Skip diagonal cells (except destination)
                        if ni == nj and ni != n-1:
                            continue
                        # Only consider cells in upper-right triangle
                        if ni + nj >= n-1:
                            dp2[ni][nj] = max(dp2[ni][nj], dp2[i][j] + fruits[ni][nj])
        
        child2_fruits = dp2[n-1][n-1] - fruits[n-1][n-1]  # Subtract destination
        
        # For child 3 starting at (n-1, 0), we need to find the best path
        # Child 3 can move: up-right, right, down-right
        # Valid moves from (i,j): (i-1,j+1), (i,j+1), (i+1,j+1)
        
        # DP for child 3 - we only need to consider the lower-left triangle
        dp3 = [[-1] * n for _ in range(n)]
        dp3[n-1][0] = fruits[n-1][0]
        
        for j in range(n-1):
            for i in range(n):
                if dp3[i][j] == -1:
                    continue
                # Try all three moves
                for di in [-1, 0, 1]:
                    ni, nj = i + di, j + 1
                    if 0 <= ni < n and nj < n:
                        # Skip diagonal cells (except destination)
                        if ni == nj and ni != n-1:
                            continue
                        # Only consider cells in lower-left triangle
                        if ni + nj >= n-1:
                            dp3[ni][nj] = max(dp3[ni][nj], dp3[i][j] + fruits[ni][nj])
        
        child3_fruits = dp3[n-1][n-1] - fruits[n-1][n-1]  # Subtract destination
        
        return child1_fruits + child2_fruits + child3_fruits