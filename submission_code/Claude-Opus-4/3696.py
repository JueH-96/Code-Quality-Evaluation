class Solution:
    def countSubstrings(self, s: str) -> int:
        count = 0
        n = len(s)
        
        for i in range(n):
            # Skip if current digit is 0 (can't divide by 0)
            if s[i] == '0':
                continue
                
            last_digit = int(s[i])
            num = 0
            power = 1
            
            # Build number from right to left
            for j in range(i, -1, -1):
                num = int(s[j]) * power + num
                power *= 10
                
                # Check if the number is divisible by its last digit
                if num % last_digit == 0:
                    count += 1
        
        return count