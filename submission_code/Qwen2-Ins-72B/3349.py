class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        left = 0
        right = 0
        max_length = 0
        char_count = {}
        
        while right < len(s):
            char = s[right]
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
            
            while char_count[char] > 2:
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1
            
            max_length = max(max_length, right - left + 1)
            right += 1
        
        return max_length