class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = []
        n = len(s)
        
        # Process each substring of length k
        for i in range(0, n, k):
            substring = s[i:i+k]
            
            # Calculate sum of hash values
            hash_sum = 0
            for char in substring:
                hash_sum += ord(char) - ord('a')
            
            # Get the hashed character
            hashed_char = hash_sum % 26
            
            # Convert back to character and append
            result.append(chr(hashed_char + ord('a')))
        
        return ''.join(result)