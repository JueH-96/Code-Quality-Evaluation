class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        i = 0
        n = len(word)
        
        while i < n:
            c = word[i]
            count = 0
            
            # Count the number of repeating characters
            while i < n and word[i] == c and count < 9:
                count += 1
                i += 1
            
            # Append the count and character to the compressed string
            comp += str(count) + c
            
            # If there are more than 9 of the same character, continue counting
            while i < n and word[i] == c:
                count = 0
                while i < n and word[i] == c and count < 9:
                    count += 1
                    i += 1
                comp += str(count) + c
        
        return comp