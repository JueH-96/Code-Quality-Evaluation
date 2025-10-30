class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = []
        for i in range(0, len(s), k):
            chunk = s[i:i+k]
            total = sum(ord(c) - ord('a') for c in chunk)
            hashed = total % 26
            result.append(chr(hashed + ord('a')))
        return ''.join(result)