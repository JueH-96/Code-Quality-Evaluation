class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        max_length = 0
        start = 0
        char_count = {}
        
        for end in range(len(s)):
            char_count[s[end]] = char_count.get(s[end], 0) + 1
            
            while any(count > 2 for count in char_count.values()):
                char_count[s[start]] -= 1
                if char_count[s[start]] == 0:
                    del char_count[s[start]]
                start += 1
            
            max_length = max(max_length, end - start + 1)
        
        return max_length