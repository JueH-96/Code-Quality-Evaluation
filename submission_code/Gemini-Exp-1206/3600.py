class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        length = 1
        operation_count = 0

        while length < k:
            new_word = ""
            for char in word:
                new_char = chr(((ord(char) - ord('a') + 1) % 26) + ord('a'))
                new_word += new_char
            
            word += new_word
            length = len(word)
            operation_count +=1

        return word[k-1]