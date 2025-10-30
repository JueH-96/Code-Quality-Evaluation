class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        i = 0
        
        while i < len(word):
            char = word[i]
            count = 1
            
            # Count consecutive occurrences of the same character, up to 9
            while i + count < len(word) and word[i + count] == char and count < 9:
                count += 1
            
            # Append count and character to comp
            comp += str(count) + char
            
            # Move the index forward
            i += count
        
        return comp