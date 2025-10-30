class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        
        # Get the character count of word2
        word2_count = Counter(word2)
        n = len(word1)
        valid_count = 0
        
        # Check all possible substrings of word1
        for i in range(n):
            substr_count = Counter()
            for j in range(i, n):
                # Add the current character to our substring counter
                substr_count[word1[j]] += 1
                
                # Check if this substring is valid
                # A substring is valid if it has at least as many occurrences
                # of each character as word2 does
                if all(substr_count[char] >= count for char, count in word2_count.items()):
                    valid_count += 1
        
        return valid_count