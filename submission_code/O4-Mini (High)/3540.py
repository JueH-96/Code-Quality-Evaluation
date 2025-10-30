class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result_chars = []
        # Process each chunk of length k
        for i in range(0, len(s), k):
            chunk = s[i:i+k]
            # Sum the alphabet indices for the chunk
            total = sum(ord(c) - ord('a') for c in chunk)
            # Compute the hashed character and append to result
            hashed_char = chr((total % 26) + ord('a'))
            result_chars.append(hashed_char)
        return ''.join(result_chars)