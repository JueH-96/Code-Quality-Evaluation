class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = []
        n = len(s)
        for i in range(0, n, k):
            chunk = s[i:i+k]
            total = 0
            for c in chunk:
                total += ord(c) - ord('a')
            mod = total % 26
            result.append(chr(ord('a') + mod))
        return ''.join(result)