class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        left = 0
        zeros = 0
        ones = 0
        
        for right in range(n):
            # Expand window by including s[right]
            if s[right] == '0':
                zeros += 1
            else:
                ones += 1
            
            # Shrink window from left while constraint is violated
            # Constraint is violated when BOTH zeros > k AND ones > k
            while zeros > k and ones > k:
                if s[left] == '0':
                    zeros -= 1
                else:
                    ones -= 1
                left += 1
            
            # All substrings ending at right and starting from left to right are valid
            count += right - left + 1
        
        return count