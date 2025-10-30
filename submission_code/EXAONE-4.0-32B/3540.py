class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result_chars = []
        n = len(s)
        num_segments = n // k
        for i in range(num_segments):
            start_index = i * k
            end_index = start_index + k
            substring = s[start_index:end_index]
            total = 0
            for char in substring:
                total += ord(char) - ord('a')
            hashed_char_index = total % 26
            result_chars.append(chr(hashed_char_index + ord('a')))
        return ''.join(result_chars)