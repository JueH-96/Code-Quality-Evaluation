class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) < k:
            new_str = ''.join([chr(ord(c) + 1) for c in word])
            word += new_str
        return word[k-1]