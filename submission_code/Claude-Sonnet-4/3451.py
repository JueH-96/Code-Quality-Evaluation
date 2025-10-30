class Solution:
    def compressedString(self, word: str) -> str:
        if not word:
            return ""
        
        comp = ""
        i = 0
        
        while i < len(word):
            current_char = word[i]
            count = 1
            
            # Count consecutive occurrences of the same character, up to 9
            while i + count < len(word) and word[i + count] == current_char and count < 9:
                count += 1
            
            # Append count and character to result
            comp += str(count) + current_char
            
            # Move to the next unprocessed character
            i += count
        
        return comp