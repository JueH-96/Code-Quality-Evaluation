class Solution:
    def kthCharacter(self, k: int) -> str:
        # Start with the initial word
        word = "a"
        
        # Keep performing the operation until the word is at least k characters long
        while len(word) < k:
            # Generate the next part by shifting each character
            next_part = []
            for ch in word:
                if ch == 'z':
                    next_part.append('a')
                else:
                    next_part.append(chr(ord(ch) + 1))
            # Append it to the original word
            word += ''.join(next_part)
        
        # Return the k-th character (1-based index)
        return word[k - 1]