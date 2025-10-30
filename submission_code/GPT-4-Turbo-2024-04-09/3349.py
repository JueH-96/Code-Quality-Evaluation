class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        max_len = 0
        left = 0
        char_count = {}
        
        for right in range(len(s)):
            char = s[right]
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
            
            # Ensure no character count exceeds 2
            while any(count > 2 for count in char_count.values()):
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1
            
            # Update the maximum length found
            max_len = max(max_len, right - left + 1)
        
        return max_len