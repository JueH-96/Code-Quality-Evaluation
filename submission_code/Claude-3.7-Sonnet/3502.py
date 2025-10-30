class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        
        for i in range(n):
            char_count = {}
            
            for j in range(i, n):
                char = s[j]
                char_count[char] = char_count.get(char, 0) + 1
                
                # Check if any character appears at least k times
                for c, freq in char_count.items():
                    if freq >= k:
                        count += 1
                        break
        
        return count