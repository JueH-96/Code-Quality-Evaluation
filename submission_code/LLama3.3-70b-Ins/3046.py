class Solution:
    def minimumOperations(self, num: str) -> int:
        memo = {}
        
        def dfs(index, num_str):
            if num_str in memo:
                return memo[num_str]
            
            if int(num_str) % 25 == 0:
                return 0
            
            min_ops = float('inf')
            for i in range(len(num_str)):
                new_str = num_str[:i] + num_str[i+1:]
                if new_str and new_str[0] != '0' or not new_str:
                    min_ops = min(min_ops, 1 + dfs(index, new_str))
            
            memo[num_str] = min_ops
            return min_ops
        
        return dfs(0, num)