class Solution:
    def kthCharacter(self, k: int) -> str:
        t = bin(k - 1).count('1')
        return chr(ord('a') + (t % 26))