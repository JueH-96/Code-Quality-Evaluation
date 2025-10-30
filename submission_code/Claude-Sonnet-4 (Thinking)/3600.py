class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        
        while len(word) < k:
            # Generate new string by changing each character to its next character
            new_string = ""
            for char in word:
                if char == 'z':
                    new_string += 'a'
                else:
                    new_string += chr(ord(char) + 1)
            
            # Append to original word
            word += new_string
        
        return word[k-1]