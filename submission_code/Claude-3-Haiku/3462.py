class Solution:
    def doesAliceWin(self, s: str) -> bool:
        def count_vowels(substring):
            vowels = set('aeiou')
            count = 0
            for char in substring:
                if char in vowels:
                    count += 1
            return count
        
        def dfs(s):
            if not s:
                return False
            
            for i in range(1, len(s)+1):
                substring = s[:i]
                if count_vowels(substring) % 2 != 0:
                    if not dfs(s[i:]):
                        return True
            
            return False
        
        return dfs(s)