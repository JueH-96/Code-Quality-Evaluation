class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        M, N = len(grid), len(grid[0])
        ans = 0
        dp = [[0] * N for _ in range(M)]
        
        for i in range(M):
            for j in range(N):
                if grid[i][j] == "X":
                    dp[i][j] = 1 if j == 0 else dp[i][j - 1] + 1
                    balance, cnt = 1, 0
                    for k in range(j, -1, -1):
                        balance += 1 if grid[i][k] == "X" else -1
                        if balance == 0:
                            cnt += 1
                    for k in range(i + 1, M):
                        if grid[k][j] != "X":
                            break
                        balance = 1
                        for l in range(j, -1, -1):
                            balance += 1 if grid[k][l] == "X" else -1
                            if balance == 0:
                                cnt += 1
                    ans += cnt
        return ans