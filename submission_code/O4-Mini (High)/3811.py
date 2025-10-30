class Solution:
    def reverseDegree(self, s: str) -> int:
        total = 0
        for idx, ch in enumerate(s, start=1):
            # 'a' -> 26, 'b' -> 25, ..., 'z' -> 1
            rev_pos = 26 - (ord(ch) - ord('a'))
            total += rev_pos * idx
        return total