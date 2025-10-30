class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = []
        for i in range(0, len(s), k):
            sub = s[i:i+k]
            sum_hash = sum(ord(c) - ord('a') for c in sub)
            hashed_char = sum_hash % 26
            result.append(chr(hashed_char + ord('a')))
        return "".join(result)