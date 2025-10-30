class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        length = 1
        iteration = 0
        while length < k:
            iteration +=1
            next_str = ""
            for char in word:
                next_char = chr(((ord(char) - ord('a') + 1) % 26) + ord('a'))
                next_str += next_char
            word += next_str
            length += len(next_str)

        return word[k-1]