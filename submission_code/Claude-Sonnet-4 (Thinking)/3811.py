class Solution:
    def reverseDegree(self, s: str) -> int:
        total = 0
        for i, ch in enumerate(s):
            reversed_alphabet_pos = ord('z') - ord(ch) + 1
            string_pos = i + 1  # 1-indexed position
            total += reversed_alphabet_pos * string_pos
        return total