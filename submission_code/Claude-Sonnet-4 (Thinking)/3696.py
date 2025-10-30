class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        
        for i in range(n):
            if s[i] == '0':
                continue  # Can't divide by 0
            
            d = int(s[i])
            current_val_mod = 0
            multiplier = 1
            
            # Build substrings ending at position i, going backwards
            for j in range(i, -1, -1):
                # Add the digit s[j] with appropriate power of 10
                current_val_mod = (int(s[j]) * multiplier + current_val_mod) % d
                if current_val_mod == 0:
                    count += 1
                # Update multiplier for next iteration (next power of 10)
                multiplier = (multiplier * 10) % d
        
        return count