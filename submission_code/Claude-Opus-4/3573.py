class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        
        # Count frequency of characters in word2
        word2_count = Counter(word2)
        required_chars = len(word2_count)  # Number of unique characters we need
        
        # Sliding window variables
        window_count = Counter()
        formed_chars = 0  # Number of unique characters in window with required frequency
        left = 0
        result = 0
        
        # Expand window with right pointer
        for right in range(len(word1)):
            char = word1[right]
            window_count[char] += 1
            
            # Check if this character now meets the requirement
            if char in word2_count and window_count[char] == word2_count[char]:
                formed_chars += 1
            
            # Contract window from left while it's still valid
            while formed_chars == required_chars and left <= right:
                # All substrings from current left to any position >= right are valid
                result += len(word1) - right
                
                # Remove leftmost character
                left_char = word1[left]
                window_count[left_char] -= 1
                if left_char in word2_count and window_count[left_char] < word2_count[left_char]:
                    formed_chars -= 1
                left += 1
        
        return result