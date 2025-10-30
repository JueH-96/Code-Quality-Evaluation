class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        
        for j in range(n):
            last_digit = int(s[j])
            if last_digit == 0:  # Skip if last digit is 0
                continue
            
            for i in range(j + 1):
                # Calculate if substring s[i:j+1] is divisible by last_digit
                remainder = 0
                for k in range(i, j + 1):
                    remainder = (remainder * 10 + int(s[k])) % last_digit
                
                if remainder == 0:
                    count += 1
        
        return count