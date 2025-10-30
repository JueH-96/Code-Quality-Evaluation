class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def dfs(i, mask, changed):
            if i == len(s):
                return 1
            
            key = (i, mask, changed)
            if key in memo:
                return memo[key]
            
            curr_char = ord(s[i]) - ord('a')
            new_mask = mask | (1 << curr_char)
            distinct = bin(new_mask).count('1')
            
            if distinct <= k:
                result = dfs(i + 1, new_mask, changed)
            else:
                result = 1 + dfs(i + 1, 1 << curr_char, changed)
            
            if not changed:
                for j in range(26):
                    alt_mask = mask | (1 << j)
                    alt_distinct = bin(alt_mask).count('1')
                    
                    if alt_distinct <= k:
                        result = max(result, dfs(i + 1, alt_mask, True))
                    else:
                        result = max(result, 1 + dfs(i + 1, 1 << j, True))
            
            memo[key] = result
            return result
        
        memo = {}
        return dfs(0, 0, False)