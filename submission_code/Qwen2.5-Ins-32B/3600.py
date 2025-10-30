class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) < k:
            new_str = ""
            for char in word:
                if char == 'z':
                    new_str += 'a'
                else:
                    new_str += chr(ord(char) + 1)
            word += new_str
        return word[k-1]