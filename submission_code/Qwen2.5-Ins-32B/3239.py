class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        memo = {}
        
        def dfs(x):
            if x == y:
                return 0
            if x < y:
                return y - x
            if x in memo:
                return memo[x]
            
            ops = float('inf')
            if x % 11 == 0:
                ops = min(ops, 1 + dfs(x // 11))
            if x % 5 == 0:
                ops = min(ops, 1 + dfs(x // 5))
            ops = min(ops, 1 + dfs(x - 1))
            ops = min(ops, 1 + dfs(x + 1))
            
            memo[x] = ops
            return ops
        
        return dfs(x)