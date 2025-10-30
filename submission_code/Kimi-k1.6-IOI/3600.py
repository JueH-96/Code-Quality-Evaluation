class Solution:
    def kthCharacter(self, k: int) -> str:
        count = bin(k - 1).count('1')
        return chr(ord('a') + count % 26)