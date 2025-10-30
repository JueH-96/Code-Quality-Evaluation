class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        max_len = 0
        n = len(s)
        
        for i in range(n):
            char_count = {}
            curr_len = 0
            
            for j in range(i, n):
                char = s[j]
                char_count[char] = char_count.get(char, 0) + 1
                
                if char_count[char] > 2:
                    break
                    
                curr_len += 1
                max_len = max(max_len, curr_len)
                
        return max_len