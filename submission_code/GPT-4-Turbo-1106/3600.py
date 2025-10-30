class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) < k:
            # Generate the new string by changing each character to its next character
            next_string = ''.join(chr(((ord(c) - ord('a') + 1) % 26) + ord('a')) for c in word)
            # Append the new string to the original word
            word += next_string
        # Return the k-th character
        return word[k - 1]