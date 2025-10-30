class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        result = 0
        
        for i in range(n):
            char_count = {}
            for j in range(i, n):
                char_count[s[j]] = char_count.get(s[j], 0) + 1
                if char_count[s[j]] == k:
                    result += 1
        
        return result