class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        n = len(s)
        
        # Process each substring of length k
        for i in range(0, n, k):
            # Get current substring
            substring = s[i:i+k]
            
            # Calculate sum of hash values
            hash_sum = 0
            for c in substring:
                hash_sum += ord(c) - ord('a')
            
            # Get remainder and corresponding character
            hashed_char = hash_sum % 26
            result += chr(ord('a') + hashed_char)
            
        return result