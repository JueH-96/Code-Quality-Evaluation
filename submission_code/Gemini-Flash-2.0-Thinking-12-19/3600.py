class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) < k:
            generated_string = ""
            for char in word:
                if char == 'z':
                    generated_string += 'a'
                else:
                    generated_string += chr(ord(char) + 1)
            word += generated_string
        return word[k-1]