class Solution:
    def compressedString(self, word: str) -> str:
        if not word:
            return ""
        
        comp = []
        i = 0
        
        while i < len(word):
            # Find the current character and count consecutive occurrences
            current_char = word[i]
            count = 0
            j = i
            
            # Count how many consecutive characters we have
            while j < len(word) and word[j] == current_char:
                count += 1
                j += 1
            
            # Split the count into chunks of at most 9
            while count > 0:
                chunk_size = min(count, 9)
                comp.append(str(chunk_size) + current_char)
                count -= chunk_size
            
            # Move to the next different character
            i = j
        
        return ''.join(comp)