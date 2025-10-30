class Solution:
    def kthCharacter(self, k: int) -> str:
        if k == 1:
            return 'a'
        m = 0
        current = 1
        while current < k:
            current <<= 1
            m += 1
        half = 1 << (m - 1)
        if k <= half:
            return self.kthCharacter(k)
        else:
            prev_char = self.kthCharacter(k - half)
            return self.next_char(prev_char)
    
    def next_char(self, c: str) -> str:
        return chr((ord(c) - ord('a') + 1) % 26 + ord('a'))