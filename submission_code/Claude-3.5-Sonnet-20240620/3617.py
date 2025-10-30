class Solution:
    def possibleStringCount(self, word: str) -> int:
        n = len(word)
        count = 1  # Initialize count with 1 for the original string
        
        # Check for consecutive repeated characters
        for i in range(n - 1):
            if word[i] == word[i + 1]:
                # Found a repeated character, count possible strings
                left = word[:i + 1]
                right = word[i + 2:]
                
                # Count strings with one less repetition
                count += 1
                
                # Count strings with repetition removed completely
                if i == 0 or word[i - 1] != word[i]:
                    count += 1
        
        return count