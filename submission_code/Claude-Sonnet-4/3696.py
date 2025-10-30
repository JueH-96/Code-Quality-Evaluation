class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        
        for end in range(n):
            for start in range(end + 1):
                substring = s[start:end + 1]
                
                # Find the last non-zero digit
                last_nonzero_digit = None
                for i in range(len(substring) - 1, -1, -1):
                    if substring[i] != '0':
                        last_nonzero_digit = int(substring[i])
                        break
                
                # If no non-zero digit found, skip
                if last_nonzero_digit is None:
                    continue
                
                # Check if the number is divisible by the last non-zero digit
                num = int(substring)
                if num % last_nonzero_digit == 0:
                    count += 1
        
        return count