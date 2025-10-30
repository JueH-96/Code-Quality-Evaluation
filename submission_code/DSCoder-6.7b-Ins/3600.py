class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        i = 0
        while len(word) < k:
            new_word = ""
            for char in word:
                if char == 'z':
                    new_word += 'a'
                else:
                    new_word += chr(ord(char) + 1)
            word += new_word
        return word[k-1]