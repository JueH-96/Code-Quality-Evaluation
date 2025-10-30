class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        
        need = Counter(word2)
        required = len(need)
        
        left = 0
        formed = 0
        window_counts = {}
        result = 0
        
        for right in range(len(word1)):
            # Expand window
            char = word1[right]
            window_counts[char] = window_counts.get(char, 0) + 1
            
            if char in need and window_counts[char] == need[char]:
                formed += 1
            
            # Contract window while it's valid
            while formed == required:
                # All substrings ending at 'right' and starting from index 0 to 'left' are valid
                result += left + 1
                
                # Remove leftmost character
                char = word1[left]
                window_counts[char] -= 1
                if char in need and window_counts[char] < need[char]:
                    formed -= 1
                left += 1
        
        return result