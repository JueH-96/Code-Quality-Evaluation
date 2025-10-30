class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) < k:
            # Generate the new string by shifting each character to its next in alphabet
            new_str = ''.join(chr((ord(c) - ord('a') + 1) % 26 + ord('a')) for c in word)
            word += new_str  # Append the new string to the current word
        return word[k-1]