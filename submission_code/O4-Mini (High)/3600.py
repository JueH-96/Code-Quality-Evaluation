class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        # Keep building until we have at least k characters
        while len(word) < k:
            # Generate the next string by shifting each character
            shifted = []
            for ch in word:
                # Compute next character with wrap from 'z' to 'a'
                if ch == 'z':
                    shifted.append('a')
                else:
                    shifted.append(chr(ord(ch) + 1))
            # Append the shifted string
            word += ''.join(shifted)
        # Return the k-th character (1-indexed)
        return word[k-1]