class Solution:
    def kthCharacter(self, k: int) -> str:
        if k == 1:
            return 'a'
        m = 1
        while m * 2 < k:
            m *= 2
        offset = k - m
        prev_char = self.kthCharacter(offset)
        return chr((ord(prev_char) - ord('a') + 1) % 26 + ord('a'))