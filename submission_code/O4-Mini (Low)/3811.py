class Solution:
    def reverseDegree(self, s: str) -> int:
        total = 0
        for idx, ch in enumerate(s, start=1):
            # reversed alphabet position: 'a' -> 26, 'b' -> 25, ..., 'z' -> 1
            rev_pos = ord('z') - ord(ch) + 1
            total += idx * rev_pos
        return total