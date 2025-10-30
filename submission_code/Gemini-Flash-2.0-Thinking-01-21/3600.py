class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        current_length = 1
        while current_length < k:
            generated_string = ""
            for char in word:
                next_char = chr(((ord(char) - ord('a') + 1) % 26) + ord('a'))
                generated_string += next_char
            word += generated_string
            current_length = len(word)
        return word[k-1]