class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        d = {}
        for ic in range(len(coordinates)):
            d[coordinates[ic][0], coordinates[ic][1]] = ic
        X, Y = coordinates[k][0], coordinates[k][1]
        dp = [1 for _ in range(len(coordinates))]
        def dfs(x: int, y: int) -> int:
            ans = 0
            nonlocal d, dp
            if x > X and y > Y and (x, y) in d:
                if dp[d[x, y]] != 1:
                    return dp[d[x, y]]
                ans = dfs(x, y) + 1
            if (x, y) not in d:
                return float('-inf')
            dp[d[x, y]] = ans
            return dp[d[x, y]]
        ans = 0
        for ic in range(len(coordinates)):
            ans = max(ans, dfs(coordinates[ic][0], coordinates[ic][1]))
        return ans