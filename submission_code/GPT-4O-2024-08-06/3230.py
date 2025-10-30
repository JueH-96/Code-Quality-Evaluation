class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        if n < 2:
            return 0
        
        operations = 0
        i = 0
        
        while i < n - 1:
            if abs(ord(word[i]) - ord(word[i + 1])) <= 1:
                # Find the next character that is not almost-equal
                j = i + 1
                while j < n and abs(ord(word[i]) - ord(word[j])) <= 1:
                    j += 1
                # We need to change one character in the sequence
                operations += 1
                # Move i to the next character after the sequence
                i = j
            else:
                i += 1
        
        return operations