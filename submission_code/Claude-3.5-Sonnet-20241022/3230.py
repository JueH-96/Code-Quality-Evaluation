class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        operations = 0
        i = 1
        
        while i < n:
            # Check if current character is almost equal to previous character
            if abs(ord(word[i]) - ord(word[i-1])) <= 1:
                operations += 1
                i += 2  # Skip next character since we modified current one
            else:
                i += 1
                
        return operations