class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        n = len(s)
        
        # Process each substring of length k
        for i in range(0, n, k):
            substring = s[i:i+k]
            hash_sum = 0
            
            # Calculate sum of hash values for characters in substring
            for char in substring:
                hash_sum += ord(char) - ord('a')
            
            # Find remainder when divided by 26 and convert to character
            hashed_char = hash_sum % 26
            result += chr(hashed_char + ord('a'))
        
        return result