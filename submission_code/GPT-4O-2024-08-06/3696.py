class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        
        # Iterate over all possible substrings
        for i in range(n):
            num = 0
            for j in range(i, n):
                num = num * 10 + int(s[j])  # Construct the number
                last_digit = int(s[j])
                
                # Check if the last digit is non-zero and num is divisible by last_digit
                if last_digit != 0 and num % last_digit == 0:
                    count += 1
        
        return count