class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        
        # Count frequency of characters in word2
        word2_count = Counter(word2)
        required_chars = len(word2_count)
        
        n = len(word1)
        total_count = 0
        
        for i in range(n):
            window_count = Counter()
            formed_chars = 0
            
            for j in range(i, n):
                # Add current character to window
                char = word1[j]
                window_count[char] += 1
                
                # Check if this character requirement is satisfied
                if char in word2_count and window_count[char] == word2_count[char]:
                    formed_chars += 1
                
                # If all character requirements are satisfied
                if formed_chars == required_chars:
                    # All substrings from i to j, j+1, ..., n-1 are valid
                    total_count += n - j
                    break
        
        return total_count