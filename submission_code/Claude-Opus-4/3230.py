class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        if n <= 1:
            return 0
        
        word_list = list(word)
        operations = 0
        
        i = 0
        while i < n - 1:
            # Check if current and next characters are almost-equal
            if self.areAlmostEqual(word_list[i], word_list[i + 1]):
                operations += 1
                # Skip the next character since we're changing it
                i += 2
            else:
                i += 1
        
        return operations
    
    def areAlmostEqual(self, a: str, b: str) -> bool:
        # Two characters are almost-equal if they are the same or adjacent in alphabet
        return a == b or abs(ord(a) - ord(b)) == 1