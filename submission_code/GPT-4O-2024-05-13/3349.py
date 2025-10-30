class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        from collections import defaultdict
        
        max_len = 0
        start = 0
        char_count = defaultdict(int)
        
        for end in range(len(s)):
            char_count[s[end]] += 1
            
            while any(count > 2 for count in char_count.values()):
                char_count[s[start]] -= 1
                if char_count[s[start]] == 0:
                    del char_count[s[start]]
                start += 1
            
            max_len = max(max_len, end - start + 1)
        
        return max_len