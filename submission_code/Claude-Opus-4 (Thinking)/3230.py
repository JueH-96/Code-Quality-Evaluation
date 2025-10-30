class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        operations = 0
        i = 0
        
        while i < n - 1:
            # Check if current and next characters are almost-equal
            if abs(ord(word[i]) - ord(word[i+1])) <= 1:
                operations += 1
                # Skip the next character since we would change it
                i += 2
            else:
                i += 1
        
        return operations