class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        while len(word) < k:
            new_part = ''.join(chr((ord(c) - 96) % 26 + 97) for c in word)
            word += new_part
        return word[k-1]