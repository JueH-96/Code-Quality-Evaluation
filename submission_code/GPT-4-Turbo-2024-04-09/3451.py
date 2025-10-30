class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        i = 0
        n = len(word)
        
        while i < n:
            current_char = word[i]
            start = i
            
            # Move i to count occurrences of the same character
            while i < n and word[i] == current_char:
                i += 1
            
            count = i - start
            
            # Append the count in chunks of at most 9
            while count > 9:
                comp += "9" + current_char
                count -= 9
            comp += str(count) + current_char
        
        return comp