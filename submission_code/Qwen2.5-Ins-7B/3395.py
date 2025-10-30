class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        
        count_s = Counter(s)
        unique_chars = set(s)
        
        for char in unique_chars:
            count_s[char] %= s.count(char)
            if count_s[char] == 0:
                unique_chars.remove(char)
        
        return len(unique_chars)