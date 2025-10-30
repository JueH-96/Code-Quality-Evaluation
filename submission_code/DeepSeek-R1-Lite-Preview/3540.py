class Solution:
    def stringHash(self, s: str, k: int) -> str:
        result = ""
        num_substrings = len(s) // k  # Since n is divisible by k
        
        for i in range(num_substrings):
            substring = s[i*k : (i+1)*k]
            sum_hash = sum(ord(char) - ord('a') for char in substring)
            hashedChar = sum_hash % 26
            result_char = chr(hashedChar + ord('a'))
            result += result_char
        
        return result