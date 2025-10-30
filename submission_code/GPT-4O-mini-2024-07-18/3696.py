class Solution:
    def countSubstrings(self, s: str) -> int:
        count = 0
        n = len(s)
        
        # Iterate over all possible substrings
        for i in range(n):
            # We only care about substrings ending with a non-zero digit
            if s[i] == '0':
                continue
            
            # Convert the current substring to an integer
            current_number = 0
            
            for j in range(i, n):
                current_number = current_number * 10 + int(s[j])
                
                # Check if the current number is divisible by its last digit
                if current_number % int(s[j]) == 0:
                    count += 1
        
        return count