class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        n = len(s)
        
        # Process each substring of length k
        for i in range(0, n, k):
            substring = s[i:i+k]
            
            # Calculate sum of hash values for this substring
            hash_sum = 0
            for char in substring:
                hash_sum += ord(char) - ord('a')
            
            # Get the hashed character
            hashed_char_index = hash_sum % 26
            hashed_char = chr(ord('a') + hashed_char_index)
            
            # Append to result
            result += hashed_char
        
        return result