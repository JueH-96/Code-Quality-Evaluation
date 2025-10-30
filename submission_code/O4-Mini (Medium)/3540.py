class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = []
        # Process each substring of length k
        for i in range(0, len(s), k):
            substring = s[i:i+k]
            # Sum hash values of characters: 'a'->0, 'b'->1, ..., 'z'->25
            total = sum((ord(c) - ord('a')) for c in substring)
            # Take modulo 26 and map back to a character
            hashed_char = chr((total % 26) + ord('a'))
            result.append(hashed_char)
        return "".join(result)