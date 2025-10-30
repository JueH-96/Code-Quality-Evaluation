class Solution:
    def reverseDegree(self, s: str) -> int:
        result = 0
        for i, ch in enumerate(s):
            # Position in reversed alphabet: 'a'=26, 'b'=25, ..., 'z'=1
            reversed_pos = ord('z') - ord(ch) + 1
            # Position in string (1-indexed)
            string_pos = i + 1
            # Add product to result
            result += reversed_pos * string_pos
        return result