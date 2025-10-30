class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones = [i for i, num in enumerate(nums) if num == 1]
        m = len(ones)
        cost = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                mid = (i + j) // 2
                c = 0
                for l in range(i, j + 1):
                    c += abs(ones[l] - ones[mid])
                cost[i][j] = c

        dp = [[float('inf')] * (maxChanges + 1) for _ in range(k + 1)]
        dp[0][0] = 0
        for i in range(1, k + 1):
            for j in range(maxChanges + 1):
                for l in range(i):
                    changes = 0
                    for p in range(l, i):
                        if nums[ones[(p + i -1 ) // 2]] == 0:
                            changes +=1
                    
                    if j >= changes:
                        
                        dp[i][j] = min(dp[i][j], dp[l][j - changes] + cost[l][i - 1] if l < i else 0)

        ans = float('inf')
        for j in range(maxChanges + 1):
            ans = min(ans, dp[k][j])
        return ans