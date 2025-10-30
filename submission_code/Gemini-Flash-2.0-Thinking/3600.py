class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) < k:
            next_segment = ""
            for char in word:
                if char == 'z':
                    next_segment += 'a'
                else:
                    next_segment += chr(ord(char) + 1)
            word += next_segment
        return word[k - 1]