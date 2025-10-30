class Solution:
    def reverseDegree(self, s: str) -> int:
        total = 0
        for i, ch in enumerate(s, start=1):
            # Calculate the position in the reversed alphabet:
            # 'a' should return 26, 'b' returns 25, ..., 'z' returns 1.
            reversed_index = ord('z') - ord(ch) + 1
            total += i * reversed_index
        return total