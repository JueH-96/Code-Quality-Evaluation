class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = []
        n = len(s)
        
        # Process each substring of length k
        for i in range(0, n, k):
            substring = s[i:i+k]
            
            # Calculate sum of hash values for this substring
            hash_sum = 0
            for char in substring:
                hash_sum += ord(char) - ord('a')
            
            # Find the character corresponding to hash_sum % 26
            hashedChar = hash_sum % 26
            result.append(chr(ord('a') + hashedChar))
        
        return ''.join(result)