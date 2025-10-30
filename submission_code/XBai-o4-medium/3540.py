class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = []
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            total = 0
            for c in substring:
                total += ord(c) - ord('a')
            hashed_char = total % 26
            result.append(chr(hashed_char + ord('a')))
        return ''.join(result)