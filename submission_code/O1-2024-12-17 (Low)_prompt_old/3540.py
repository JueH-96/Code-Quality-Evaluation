class Solution:
    def stringHash(self, s: str, k: int) -> str:
        n = len(s)
        result = []
        
        for i in range(0, n, k):
            substring = s[i:i+k]
            # Compute the sum of hash values of the characters in the substring
            total_hash = sum((ord(ch) - ord('a')) for ch in substring)
            # Take modulo 26
            hashed_char = total_hash % 26
            # Convert back to a character
            hashed_char_letter = chr(hashed_char + ord('a'))
            # Append to the result list
            result.append(hashed_char_letter)
        
        return "".join(result)