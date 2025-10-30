class Solution:
    def kthCharacter(self, k: int) -> str:
        word = "a"
        for _ in range(k):
            word = "".join([chr(ord(c) + 1) for c in word])
        return word[k-1]