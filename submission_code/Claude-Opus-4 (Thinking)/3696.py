class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        
        for i in range(n):
            # remainders[d] is the remainder when the current number is divided by d
            remainders = [0] * 10
            
            for j in range(i, n):
                digit = int(s[j])
                
                # Update all remainders
                for d in range(1, 10):
                    remainders[d] = (remainders[d] * 10 + digit) % d
                
                if digit != 0 and remainders[digit] == 0:
                    count += 1
        
        return count