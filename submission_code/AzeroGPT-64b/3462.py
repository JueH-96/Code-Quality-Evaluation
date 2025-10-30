from functools import lru_cache

class Solution:
    def doesAliceWin(self, s: str) -> bool:
        
        vowels = ('a', 'e', 'i', 'o', 'u')
        n = len(s)
        
        @lru_cache(None)
        def memo_dfs(index = -2):
            
            if index == n - 1:
                return False
            
            if index == n - 2:
                if s[index + 1] in vowels:
                    return True
                else:
                    return False
            
            cnt = 0
            for i in range(index + 1, n):
                if s[i] in vowels:
                    cnt += 1
                    if cnt%2 == 1:
                        if not memo_dfs(i):
                            return True
                else:
                    if cnt%2 == 1:
                        if not memo_dfs(i - 1):
                            return True
                    
                if cnt%2 == 0 and i == n - 1:
                    return False
            
            return False
        
        return memo_dfs()