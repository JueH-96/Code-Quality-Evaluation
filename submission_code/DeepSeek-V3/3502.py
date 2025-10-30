class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total = 0
        
        for i in range(n):
            char_count = {}
            for j in range(i, n):
                char = s[j]
                char_count[char] = char_count.get(char, 0) + 1
                if any(count >= k for count in char_count.values()):
                    total += 1
        
        return total