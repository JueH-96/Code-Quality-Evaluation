class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) < k:
            new_word = ""
            for char in word:
                new_char = chr(((ord(char) - ord('a') + 1) % 26) + ord('a'))
                new_word += new_char
            word += new_word
        return word[k - 1]