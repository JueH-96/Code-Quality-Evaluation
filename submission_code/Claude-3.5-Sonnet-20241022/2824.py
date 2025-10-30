class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n with 2*n and 3*n
        concat = str(n) + str(2*n) + str(3*n)
        
        # Check if length is 9 (should contain exactly 9 digits)
        if len(concat) != 9:
            return False
            
        # Check if all digits 1-9 appear exactly once and no 0s
        digits = set(concat)
        if '0' in digits:
            return False
            
        return len(digits) == 9 and all(str(i) in digits for i in range(1,10))