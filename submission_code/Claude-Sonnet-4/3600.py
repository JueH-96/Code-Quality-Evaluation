class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        
        while len(word) < k:
            # Generate new string by incrementing each character
            new_part = ""
            for char in word:
                if char == 'z':
                    new_part += 'a'
                else:
                    new_part += chr(ord(char) + 1)
            
            # Append to original word
            word += new_part
        
        return word[k-1]  # k-th character (1-indexed)