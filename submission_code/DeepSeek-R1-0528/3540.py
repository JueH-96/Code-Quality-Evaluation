class Solution:
    def stringHash(self, s: str, k: int) -> str:
        n = len(s)
        result = []
        for i in range(0, n, k):
            chunk = s[i:i+k]
            total = 0
            for char in chunk:
                total += ord(char) - ord('a')
            hashed_value = total % 26
            result.append(chr(hashed_value + ord('a')))
        return ''.join(result)