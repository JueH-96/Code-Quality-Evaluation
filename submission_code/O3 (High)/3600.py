class Solution:
    def kthCharacter(self, k: int) -> str:
        # Start with the initial word
        word = "a"
        
        # Keep performing the operation until the word is long enough
        while len(word) < k:
            # Build the transformed (shifted) version of the current word
            shifted = ''.join(chr((ord(ch) - 97 + 1) % 26 + 97) for ch in word)
            # Append the shifted string to the original
            word += shifted
        
        # Return the k-th (1-based) character
        return word[k - 1]