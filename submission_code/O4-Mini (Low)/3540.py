class Solution:
    def stringHash(self, s: str, k: int) -> str:
        # Number of substrings
        n = len(s)
        num_subs = n // k
        
        result_chars = []
        for i in range(num_subs):
            # Extract the i-th substring of length k
            start = i * k
            substring = s[start:start + k]
            
            # Compute the sum of hash values for characters in substring
            total = 0
            for ch in substring:
                total += ord(ch) - ord('a')
            
            # Compute hashed character
            hashed_char_index = total % 26
            hashed_char = chr(hashed_char_index + ord('a'))
            
            # Append to result list
            result_chars.append(hashed_char)
        
        # Join the list into the final result string
        return ''.join(result_chars)