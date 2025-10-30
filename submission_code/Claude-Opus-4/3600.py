class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        
        # Keep performing operations until we have at least k characters
        while len(word) < k:
            # Generate new string by shifting each character
            new_part = ""
            for char in word:
                if char == 'z':
                    new_part += 'a'
                else:
                    new_part += chr(ord(char) + 1)
            word += new_part
        
        # Return the k-th character (1-indexed)
        return word[k-1]