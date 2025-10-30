class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        if n == 0:
            return 0
        
        # Initialize DP tables for each child
        dp1 = [[-float('inf')] * n for _ in range(n)]
        dp2 = [[-float('inf')] * n for _ in range(n)]
        dp3 = [[-float('inf')] * n for _ in range(n)]
        
        # Base cases
        dp1[0][0] = fruits[0][0]
        dp2[0][n-1] = fruits[0][n-1]
        dp3[n-1][0] = fruits[n-1][0]
        
        # Fill DP table for child 1 (0,0) to (n-1,n-1)
        for i in range(n):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                prev = []
                if i > 0 and j > 0:
                    prev.append(dp1[i-1][j-1])
                if i > 0:
                    prev.append(dp1[i-1][j])
                if j > 0:
                    prev.append(dp1[i][j-1])
                if prev:
                    dp1[i][j] = max(prev) + fruits[i][j]
        
        # Fill DP table for child 2 (0, n-1) to (n-1,n-1)
        for i in range(n):
            for j in range(n):
                if i == 0 and j == n-1:
                    continue
                prev_i = i - 1
                prev_js = [j + 1, j, j - 1]
                prev = []
                for pj in prev_js:
                    if 0 <= pj < n and prev_i >= 0 and dp2[prev_i][pj] != -float('inf'):
                        prev.append(dp2[prev_i][pj])
                if prev:
                    dp2[i][j] = max(prev) + fruits[i][j]
        
        # Fill DP table for child 3 (n-1, 0) to (n-1, n-1)
        for j in range(n):
            for i in range(n):
                if i == n-1 and j == 0:
                    continue
                if j == 0:
                    continue
                prev_j = j - 1
                prev_is = [i + 1, i, i - 1]
                prev = []
                for pi in prev_is:
                    if 0 <= pi < n and prev_j >= 0 and dp3[pi][prev_j] != -float('inf'):
                        prev.append(dp3[pi][prev_j])
                if prev:
                    dp3[i][j] = max(prev) + fruits[i][j]
        
        # Function to backtrack from (n-1,n-1) to start for each child
        def backtrack(dp, start_i, start_j):
            i, j = n-1, n-1
            path = set()
            path.add((i, j))
            while (i, j) != (start_i, start_j):
                possible = []
                if dp == dp1:
                    if i > 0 and j > 0 and dp[i-1][j-1] != -float('inf'):
                        possible.append((dp[i-1][j-1], i-1, j-1))
                    if i > 0 and dp[i-1][j] != -float('inf'):
                        possible.append((dp[i-1][j], i-1, j))
                    if j > 0 and dp[i][j-1] != -float('inf'):
                        possible.append((dp[i][j-1], i, j-1))
                elif dp == dp2:
                    prev_i = i - 1
                    prev_js = [j + 1, j, j - 1]
                    for pj in prev_js:
                        if 0 <= pj < n and prev_i >= 0 and dp[prev_i][pj] != -float('inf'):
                            possible.append((dp[prev_i][pj], prev_i, pj))
                elif dp == dp3:
                    prev_j = j - 1
                    prev_is = [i + 1, i, i - 1]
                    for pi in prev_is:
                        if 0 <= pi < n and prev_j >= 0 and dp[pi][prev_j] != -float('inf'):
                            possible.append((dp[pi][prev_j], pi, prev_j))
                if not possible:
                    break
                max_val = max(possible, key=lambda x: x[0])
                i, j = max_val[1], max_val[2]
                path.add((i, j))
            return path
        
        # Get paths for each child
        path1 = backtrack(dp1, 0, 0)
        path2 = backtrack(dp2, 0, n-1)
        path3 = backtrack(dp3, n-1, 0)
        
        # Calculate the sum of unique cells
        visited = set()
        total = 0
        for (i, j) in path1:
            if (i, j) not in visited:
                visited.add((i, j))
                total += fruits[i][j]
        for (i, j) in path2:
            if (i, j) not in visited:
                visited.add((i, j))
                total += fruits[i][j]
        for (i, j) in path3:
            if (i, j) not in visited:
                visited.add((i, j))
                total += fruits[i][j]
        
        return total