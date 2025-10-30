class Solution:
    def kthCharacter(self, k: int) -> str:
        a = ord('a') - 1
        c = 1
        while c < k:
            a = (a + 1) % 26
            c = c * 2 + 1
        for i in range(k.bit_length() - 1):
            if k & (1 << i):
                a = (a + 1) % 26
        return chr(a + ord('a'))