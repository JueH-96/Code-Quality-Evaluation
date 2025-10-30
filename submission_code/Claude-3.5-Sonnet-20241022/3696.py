class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        
        # For each possible substring
        for i in range(n):
            for j in range(i, n):
                # Get the substring
                substr = s[i:j+1]
                last_digit = int(substr[-1])
                
                # Skip if last digit is 0
                if last_digit == 0:
                    continue
                    
                # Convert substring to integer
                num = int(substr)
                
                # Check if number is divisible by last digit
                if num % last_digit == 0:
                    count += 1
        
        return count