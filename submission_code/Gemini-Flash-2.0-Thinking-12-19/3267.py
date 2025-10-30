class Solution:
    def maximumLength(self, s: str) -> int:
        max_len = -1
        for char_code in range(ord('a'), ord('z') + 1):
            char = chr(char_code)
            for length in range(len(s), 0, -1):
                sub = char * length
                count = 0
                for i in range(len(s) - length + 1):
                    if s[i:i+length] == sub:
                        count += 1
                if count >= 3:
                    max_len = max(max_len, length)
                    break
        return max_len