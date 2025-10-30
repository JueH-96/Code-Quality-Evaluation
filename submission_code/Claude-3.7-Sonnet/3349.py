class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        left = 0
        max_length = 0
        char_count = {}
        
        for right in range(len(s)):
            # Increment count of current character
            char_count[s[right]] = char_count.get(s[right], 0) + 1
            
            # While the count of the current character exceeds 2, move left pointer
            while char_count[s[right]] > 2:
                char_count[s[left]] -= 1
                left += 1
            
            # Update max_length
            max_length = max(max_length, right - left + 1)
        
        return max_length