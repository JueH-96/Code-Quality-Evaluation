class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        operations = 0
        i = 0
        
        while i < len(word) - 1:
            # Check if current character and next character are almost-equal
            if self.isAlmostEqual(word[i], word[i+1]):
                operations += 1
                i += 2  # Skip both characters (we changed the second one)
            else:
                i += 1  # Move to next character
                
        return operations
    
    def isAlmostEqual(self, a: str, b: str) -> bool:
        # Characters are almost-equal if they're the same or adjacent in alphabet
        return a == b or abs(ord(a) - ord(b)) == 1